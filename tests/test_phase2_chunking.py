import unittest

from scripts.phase2_chunking import canonicalize_articles, sentence_aligned_chunks


def whitespace_tokens(text):
    return len(text.split())


class Phase2ChunkingTests(unittest.TestCase):
    def test_sentence_aligned_chunks_never_split_sentences(self):
        article = {
            "article_id": "article_test",
            "document_title": "Title",
            "sentence_texts": ["one two three", "four five", "six seven eight"],
            "legacy_doc_ids": ["d1"],
        }
        chunks = sentence_aligned_chunks(article, 5, whitespace_tokens)
        self.assertEqual([chunk["sentence_ids"] for chunk in chunks], [[0, 1], [2]])
        self.assertEqual(chunks[0]["sentence_texts"], article["sentence_texts"][:2])
        self.assertEqual(chunks[1]["sentence_texts"], article["sentence_texts"][2:])

    def test_oversized_sentence_is_preserved_with_its_id(self):
        article = {
            "article_id": "article_test",
            "document_title": "Title",
            "sentence_texts": ["one two three four five six"],
            "legacy_doc_ids": ["d1"],
        }
        chunks = sentence_aligned_chunks(article, 4, whitespace_tokens)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["sentence_ids"], [0])
        self.assertTrue(chunks[0]["oversized_single_sentence"])

    def test_exact_duplicates_receive_one_canonical_article(self):
        instances = [
            {
                "legacy_doc_id": "d1",
                "document_title": "Same",
                "sentence_texts": ["Text."],
                "text": "Text.",
                "source_question_id": "q1",
            },
            {
                "legacy_doc_id": "d2",
                "document_title": "Same",
                "sentence_texts": ["Text."],
                "text": "Text.",
                "source_question_id": "q2",
            },
        ]
        articles, audit = canonicalize_articles(instances)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["legacy_doc_ids"], ["d1", "d2"])
        self.assertEqual(audit["duplicate_title_identical_text_extra_instances"], 1)

    def test_same_text_with_different_titles_is_not_merged(self):
        instances = [
            {
                "legacy_doc_id": "d1",
                "document_title": "A",
                "sentence_texts": ["Text."],
                "text": "Text.",
                "source_question_id": "q1",
            },
            {
                "legacy_doc_id": "d2",
                "document_title": "B",
                "sentence_texts": ["Text."],
                "text": "Text.",
                "source_question_id": "q2",
            },
        ]
        articles, audit = canonicalize_articles(instances)
        self.assertEqual(len(articles), 2)
        self.assertEqual(audit["different_title_identical_text_groups"], 1)


if __name__ == "__main__":
    unittest.main()
