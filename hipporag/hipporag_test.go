/*
 * SPDX-License-Identifier: Apache-2.0
 */

package hipporag

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestDerivedIndexDropAndBitExactRebuild(t *testing.T) {
	bundle := BuildStandardBenchmarkSuite()
	eng := NewEngine(DefaultPPRConfig())
	if err := eng.UpsertSourcesAndRebuild(bundle.Records); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	countBefore := eng.SourceCount()
	snapBefore := eng.Snapshot()
	if snapBefore.DigestSHA256 == "" || snapBefore.GraphNodes == 0 || snapBefore.GraphEdges == 0 {
		t.Fatalf("expected non-empty snapshot, got %+v", snapBefore)
	}

	req := QueryRequest{
		Query:  "Which change caused the CUDA graph memory alias?",
		Access: AccessContext{TenantID: defaultTenant},
		TopK:   5,
	}
	resBefore, err := eng.SearchHippoRAG2(req)
	if err != nil {
		t.Fatalf("SearchHippoRAG2 before drop: %v", err)
	}

	// Drop all derived indexes; canonical records must remain intact.
	eng.DropDerivedIndexes()
	if eng.HasDerivedIndexes() {
		t.Fatalf("expected HasDerivedIndexes() == false after drop")
	}
	if eng.SourceCount() != countBefore {
		t.Fatalf("canonical records altered by DropDerivedIndexes: got %d want %d", eng.SourceCount(), countBefore)
	}
	if _, err := eng.SearchHippoRAG2(req); err != ErrIndexesNotBuilt {
		t.Fatalf("expected ErrIndexesNotBuilt while dropped, got %v", err)
	}
	if _, err := eng.SearchBM25(req); err != ErrIndexesNotBuilt {
		t.Fatalf("expected ErrIndexesNotBuilt on BM25 while dropped, got %v", err)
	}
	if _, err := eng.SearchDenseRAG(req); err != ErrIndexesNotBuilt {
		t.Fatalf("expected ErrIndexesNotBuilt on DenseRAG while dropped, got %v", err)
	}

	// Rebuild from canonical records and verify bit-exact SHA-256 parity.
	eng.RebuildIndex()
	if !eng.HasDerivedIndexes() {
		t.Fatalf("expected HasDerivedIndexes() == true after rebuild")
	}
	snapAfter := eng.Snapshot()
	if snapBefore.DigestSHA256 != snapAfter.DigestSHA256 {
		t.Fatalf("non-deterministic rebuild digest: before=%s after=%s", snapBefore.DigestSHA256, snapAfter.DigestSHA256)
	}

	resAfter, err := eng.SearchHippoRAG2(req)
	if err != nil {
		t.Fatalf("SearchHippoRAG2 after rebuild: %v", err)
	}
	if len(resBefore.Hits) != len(resAfter.Hits) {
		t.Fatalf("hit count mismatch after rebuild: before=%d after=%d", len(resBefore.Hits), len(resAfter.Hits))
	}
	for i := range resBefore.Hits {
		if resBefore.Hits[i].SourceID != resAfter.Hits[i].SourceID ||
			resBefore.Hits[i].Path.Trail != resAfter.Hits[i].Path.Trail {
			t.Fatalf("hit %d differed after rebuild:\n before=%+v\n after=%+v", i, resBefore.Hits[i], resAfter.Hits[i])
		}
	}
}

func TestCanonicalRecordVersioningAndValidation(t *testing.T) {
	eng := NewEngine(DefaultPPRConfig())

	if err := eng.UpsertSource(SourceRecord{SourceID: "s1", Version: 1}); err != ErrEmptyTenant {
		t.Fatalf("expected ErrEmptyTenant, got %v", err)
	}
	if err := eng.UpsertSource(SourceRecord{TenantID: "t1", Version: 1}); err != ErrEmptySourceID {
		t.Fatalf("expected ErrEmptySourceID, got %v", err)
	}
	if err := eng.UpsertSource(SourceRecord{TenantID: "t1", SourceID: "s1", Version: 0}); err != ErrInvalidVersion {
		t.Fatalf("expected ErrInvalidVersion, got %v", err)
	}

	v1 := SourceRecord{
		SourceID:    "doc1",
		Version:     1,
		Kind:        SourceDocument,
		TenantID:    "t1",
		Title:       "Allocator Spec",
		Text:        "Initial spec v1",
		Reliability: 0.90,
	}
	v2 := SourceRecord{
		SourceID:    "doc1",
		Version:     2,
		Kind:        SourceDocument,
		TenantID:    "t1",
		Title:       "Allocator Spec",
		Text:        "Updated spec v2 with stream capture pool guard",
		Reliability: 0.98,
	}
	if err := eng.UpsertSource(v1); err != nil {
		t.Fatalf("upsert v1: %v", err)
	}
	if err := eng.UpsertSource(v2); err != nil {
		t.Fatalf("upsert v2: %v", err)
	}
	eng.RebuildIndex()

	if got := eng.SourceVersionCount("t1", "doc1"); got != 2 {
		t.Fatalf("expected 2 stored versions, got %d", got)
	}
	gotRec, ok := eng.GetSource(AccessContext{TenantID: "t1"}, "doc1")
	if !ok || gotRec.Version != 2 {
		t.Fatalf("expected latest version 2, got ok=%v rec=%+v", ok, gotRec)
	}
}

func TestStrictPreReturnTenantAndACLBridgeIsolation(t *testing.T) {
	recs, queries := BuildACLBridgeNegativeWorkload()
	eng := NewEngine(DefaultPPRConfig())
	if err := eng.UpsertSourcesAndRebuild(recs); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	// Empty tenant must be rejected before touching any index.
	if _, err := eng.SearchHippoRAG2(QueryRequest{Query: "cuda graph memory alias"}); err != ErrEmptyTenant {
		t.Fatalf("expected ErrEmptyTenant, got %v", err)
	}

	// Verify zero leaks across every adversarial bridge query on all 4 retrievers.
	searchers := []struct {
		name string
		fn   func(QueryRequest) (QueryResult, error)
	}{
		{"hipporag2", eng.SearchHippoRAG2},
		{"bm25", eng.SearchBM25},
		{"dense", eng.SearchDenseRAG},
		{"1hop", eng.SearchUnfiltered1Hop},
	}

	for _, s := range searchers {
		for _, q := range queries {
			res, err := s.fn(QueryRequest{Query: q.Query, Access: q.Access, TopK: 5})
			if err != nil {
				t.Fatalf("%s query %s failed: %v", s.name, q.ID, err)
			}
			if res.Audit.ForeignTenantNodesInspected != 0 || res.Audit.ForeignTenantEdgesTraversed != 0 {
				t.Fatalf("%s query %s touched foreign tenant graph: %+v", s.name, q.ID, res.Audit)
			}
			for _, h := range res.Hits {
				if h.TenantID != q.Access.TenantID {
					t.Fatalf("%s query %s leaked foreign tenant %s to %s", s.name, q.ID, h.TenantID, q.Access.TenantID)
				}
				if strings.HasPrefix(h.SourceID, "beta_") || strings.HasPrefix(h.SourceID, "alpha_restricted_") {
					t.Fatalf("%s query %s leaked restricted/confidential record %s", s.name, q.ID, h.SourceID)
				}
				if _, ok := eng.GetSource(q.Access, h.SourceID); !ok {
					t.Fatalf("%s query %s returned unauthorized source %s", s.name, q.ID, h.SourceID)
				}
			}
		}
	}

	// Verify authorized caller WITH sec:incident-response CAN read alpha_restricted_00
	// while still never seeing tenant-beta records.
	privRes, err := eng.SearchHippoRAG2(QueryRequest{
		Query:  "Which change caused cuda graph memory alias in stream capture kernel?",
		Access: AccessContext{TenantID: "tenant-alpha", Scopes: []string{"team:public", "sec:incident-response"}},
		TopK:   5,
	})
	if err != nil {
		t.Fatalf("privileged alpha query: %v", err)
	}
	sawRestricted := false
	for _, h := range privRes.Hits {
		if h.TenantID != "tenant-alpha" || strings.HasPrefix(h.SourceID, "beta_") {
			t.Fatalf("privileged alpha caller saw tenant-beta record: %+v", h)
		}
		if h.SourceID == "alpha_restricted_00" {
			sawRestricted = true
		}
	}
	if !sawRestricted {
		t.Fatalf("expected privileged tenant-alpha caller to reach alpha_restricted_00, got hits=%v", privRes.Hits)
	}
}

func TestCUDAGraphMemoryAliasPathReconstruction(t *testing.T) {
	bundle := BuildStandardBenchmarkSuite()
	eng := NewEngine(DefaultPPRConfig())
	if err := eng.UpsertSourcesAndRebuild(bundle.Records); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	var inc01 BenchmarkQuery
	for _, q := range bundle.Queries {
		if q.ID == "inc01" {
			inc01 = q
			break
		}
	}
	if inc01.ID == "" {
		t.Fatal("inc01 CUDA graph memory alias query missing from standard suite")
	}

	res, err := eng.SearchHippoRAG2(QueryRequest{
		Query:  inc01.Query,
		Access: inc01.Access,
		TopK:   5,
	})
	if err != nil {
		t.Fatalf("SearchHippoRAG2 inc01: %v", err)
	}

	hitSet := make(map[string]RetrievedHit)
	for _, h := range res.Hits {
		hitSet[h.SourceID] = h
	}
	for _, requiredID := range inc01.GoldSourceIDs {
		h, ok := hitSet[requiredID]
		if !ok {
			t.Fatalf("missing gold source %s in top-5 hits; got %+v", requiredID, res.Hits)
		}
		if h.Path.TargetNode == "" || h.Path.Trail == "" {
			t.Fatalf("hit %s missing reconstructed path: %+v", requiredID, h.Path)
		}
	}

	pred := extractGroundedAnswer(eng, inc01, res)
	if exactMatch(pred, inc01.GoldAnswer) != 1.0 {
		t.Fatalf("expected grounded answer %q, got %q", inc01.GoldAnswer, pred)
	}
	if evaluatePathReconstruction(inc01, res) != 1.0 {
		t.Fatalf("expected full path reconstruction accuracy 1.0 for inc01; trails=%v", res.Hits)
	}
}

func TestBoundedPPRBudgetsAndHubSuppression(t *testing.T) {
	bundle := BuildHubCollapseNegativeWorkload()
	cfg := DefaultPPRConfig()
	cfg.MaxExpandedNodes = 64
	cfg.MaxIterations = 20
	cfg.MaxHops = 4
	cfg.TopK = 5

	eng := NewEngine(cfg)
	if err := eng.UpsertSourcesAndRebuild(bundle.Records); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	for _, q := range bundle.Queries[:5] {
		res, err := eng.SearchHippoRAG2(QueryRequest{Query: q.Query, Access: q.Access, TopK: 5})
		if err != nil {
			t.Fatalf("SearchHippoRAG2 %s: %v", q.ID, err)
		}
		if res.Audit.ExpandedNodes > cfg.MaxExpandedNodes {
			t.Fatalf("ExpandedNodes=%d exceeded MaxExpandedNodes=%d", res.Audit.ExpandedNodes, cfg.MaxExpandedNodes)
		}
		if res.Audit.PPRIterations > cfg.MaxIterations {
			t.Fatalf("PPRIterations=%d exceeded MaxIterations=%d", res.Audit.PPRIterations, cfg.MaxIterations)
		}
	}

	negHub, err := runHubCollapseRegime(BenchmarkOptions{TopK: 5, TimedIterations: 2, InputPricePer1MUSD: 3.0})
	if err != nil {
		t.Fatalf("runHubCollapseRegime: %v", err)
	}
	if negHub.HippoRAG2Recall <= negHub.UnweightedRecall {
		t.Fatalf("expected HippoRAG 2 recall (%.3f) > unweighted hub-collapse recall (%.3f)",
			negHub.HippoRAG2Recall, negHub.UnweightedRecall)
	}
	if negHub.HippoRAG2Irrelevant >= negHub.UnweightedIrrelevant {
		t.Fatalf("expected HippoRAG 2 irrelevant ratio (%.3f) < unweighted hub-collapse ratio (%.3f)",
			negHub.HippoRAG2Irrelevant, negHub.UnweightedIrrelevant)
	}
}

func TestExternalDatasetLoaderHotpotQAAndMuSiQue(t *testing.T) {
	tmpDir := t.TempDir()

	// 1. HotpotQA JSON fixture
	hpPath := filepath.Join(tmpDir, "hotpot_sample.json")
	hpJSON := `[
		{
			"_id": "hp_ext_1",
			"question": "Which city hosts the consortium operating Aurora Telescope?",
			"answer": "Geneva",
			"supporting_facts": [["Aurora Telescope", 0], ["Helvetia Astro Consortium", 0]],
			"context": [
				["Aurora Telescope", ["Aurora Telescope is operated by Helvetia Astro Consortium."]],
				["Helvetia Astro Consortium", ["Helvetia Astro Consortium is headquartered in Geneva."]],
				["Distractor Telescope", ["Unrelated notes about telescopes and cities."]]
			]
		}
	]`
	if err := os.WriteFile(hpPath, []byte(hpJSON), 0o644); err != nil {
		t.Fatalf("write hotpot fixture: %v", err)
	}
	hpBundle, err := LoadExternalDataset(hpPath)
	if err != nil {
		t.Fatalf("LoadExternalDataset HotpotQA: %v", err)
	}
	if len(hpBundle.Records) != 3 || len(hpBundle.Queries) != 1 || len(hpBundle.Queries[0].GoldSourceIDs) != 2 {
		t.Fatalf("unexpected HotpotQA bundle: records=%d queries=%d gold=%v",
			len(hpBundle.Records), len(hpBundle.Queries), hpBundle.Queries[0].GoldSourceIDs)
	}

	// 2. MuSiQue JSONL fixture
	mqPath := filepath.Join(tmpDir, "musique_sample.jsonl")
	mqJSONL := `{"id":"mq_ext_1","question":"Which algorithm backs the engine used by Zenith Queue?","answer":"BLAKE3","paragraphs":[{"idx":0,"title":"Zenith Queue","paragraph_text":"Zenith Queue stores messages in Granite Store.","is_supporting":true},{"idx":1,"title":"Granite Store","paragraph_text":"Granite Store hashes frames with BLAKE3.","is_supporting":true},{"idx":2,"title":"Noise","paragraph_text":"Unrelated logging paragraph.","is_supporting":false}]}` + "\n"
	if err := os.WriteFile(mqPath, []byte(mqJSONL), 0o644); err != nil {
		t.Fatalf("write musique fixture: %v", err)
	}
	mqBundle, err := LoadExternalDataset(mqPath)
	if err != nil {
		t.Fatalf("LoadExternalDataset MuSiQue: %v", err)
	}
	if len(mqBundle.Records) != 3 || len(mqBundle.Queries) != 1 || len(mqBundle.Queries[0].GoldSourceIDs) != 2 {
		t.Fatalf("unexpected MuSiQue bundle: records=%d queries=%d", len(mqBundle.Records), len(mqBundle.Queries))
	}

	// Run full benchmark cleanly on the external fixture.
	rep, err := RunBenchmark(BenchmarkOptions{
		TopK:                3,
		TimedIterations:     2,
		ExternalDatasetPath: mqPath,
	})
	if err != nil {
		t.Fatalf("RunBenchmark on external dataset: %v", err)
	}
	if !rep.Honesty.ExternalDatasetUsed || len(rep.Systems) != 4 {
		t.Fatalf("expected external dataset flag and 4 systems, got %+v", rep.Honesty)
	}
}

func TestConcurrentReadWriteRebuildRaceSafety(t *testing.T) {
	bundle := BuildStandardBenchmarkSuite()
	eng := NewEngine(DefaultPPRConfig())
	if err := eng.UpsertSourcesAndRebuild(bundle.Records); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	var wg sync.WaitGroup
	// Concurrent readers across all 4 methods.
	for g := 0; g < 6; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 15; i++ {
				q := bundle.Queries[(id*7+i)%len(bundle.Queries)]
				req := QueryRequest{Query: q.Query, Access: q.Access, TopK: 5}
				switch (id + i) % 4 {
				case 0:
					_, _ = eng.SearchHippoRAG2(req)
				case 1:
					_, _ = eng.SearchBM25(req)
				case 2:
					_, _ = eng.SearchDenseRAG(req)
				default:
					_, _ = eng.SearchUnfiltered1Hop(req)
				}
			}
		}(g)
	}
	// Concurrent canonical writer + index rebuilder.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 4; i++ {
			_ = eng.UpsertSource(SourceRecord{
				SourceID:    "concurrent_doc",
				Version:     uint64(i + 1),
				Kind:        SourceDocument,
				TenantID:    defaultTenant,
				Title:       "Concurrent Update",
				Text:        "Versioned concurrent update verifying RWMutex safety.",
				Reliability: 0.90,
			})
			eng.RebuildIndex()
		}
	}()
	wg.Wait()
}

func TestEndToEndBenchmarkReportAndInvariants(t *testing.T) {
	rep, err := RunBenchmark(BenchmarkOptions{
		TopK:               5,
		TimedIterations:    3,
		InputPricePer1MUSD: 3.00,
	})
	if err != nil {
		t.Fatalf("RunBenchmark: %v", err)
	}
	if !rep.RebuildVerification.BitExactEquivalence || !rep.RebuildVerification.DroppedErrorEnforced || !rep.RebuildVerification.QueryOutputIdentical {
		t.Fatalf("rebuild verification failed: %+v", rep.RebuildVerification)
	}
	if !rep.CUDAScenario.AnswerMatch {
		t.Fatalf("CUDA graph memory alias scenario failed: %+v", rep.CUDAScenario)
	}
	if rep.NegACLBridge.HippoRAG2LeakedQueries != 0 || rep.NegACLBridge.UnfilteredControlLeakedQueries == 0 {
		t.Fatalf("ACL bridge negative regime failed: %+v", rep.NegACLBridge)
	}

	var bm25, hippo SystemMetrics
	for _, s := range rep.Systems {
		if s.CrossTenantLeaks != 0 {
			t.Fatalf("system %s had %d cross-tenant leaks", s.System, s.CrossTenantLeaks)
		}
		if s.System == "bm25_lexical_baseline" {
			bm25 = s
		}
		if s.System == "hipporag2_bounded_ppr" {
			hippo = s
		}
	}
	if hippo.MultiHopRecall <= bm25.MultiHopRecall {
		t.Fatalf("expected HippoRAG 2 multi-hop recall (%.3f) > BM25 (%.3f)", hippo.MultiHopRecall, bm25.MultiHopRecall)
	}
	if hippo.MultiHopTaskSuccess <= bm25.MultiHopTaskSuccess {
		t.Fatalf("expected HippoRAG 2 multi-hop task success (%.3f) > BM25 (%.3f)", hippo.MultiHopTaskSuccess, bm25.MultiHopTaskSuccess)
	}
	if md := RenderScoreTableMarkdown(rep); !strings.Contains(md, "hipporag2_bounded_ppr") {
		t.Fatalf("RenderScoreTableMarkdown missing hipporag2_bounded_ppr")
	}
	if readme := RenderReportREADME(rep); !strings.Contains(readme, "MIT License") {
		t.Fatalf("RenderReportREADME missing MIT license audit")
	}
}
