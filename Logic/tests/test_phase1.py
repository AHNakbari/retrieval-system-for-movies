from Logic.tests import test_crawler, test_LSH, test_preprocess, test_index, \
    test_snippet, test_spell_correction, test_evaluation

print("=" * 40)
print("=" * 40)
print("Crawler: Test the Crawler data")
test_crawler.run_crawler_test()

print("=" * 40)
print("=" * 40)
print("LSH: Test the accuracy of the LSH")
test_LSH.run_LSH_test()

print("=" * 40)
print("=" * 40)
print("Preprocess: Test the Preprocess")
test_preprocess.run_test_preprocess()

print("=" * 40)
print("=" * 40)
print("Index: Test the correctness of indexes")
test_index.run_test_index("../core/indexer/index/")

print("=" * 40)
print("=" * 40)
print("Snippet: Test the snippet quality")
test_snippet.run_test_snippet()

print("=" * 40)
print("=" * 40)
print("Spell correction: Test the spell correction quality")
test_spell_correction.test_spell_correction()

print("=" * 40)
print("=" * 40)
print("Evaluation: Test the evaluation to evaluate the IR")
test_evaluation.TestEvaluation()
