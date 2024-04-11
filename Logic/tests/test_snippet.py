from Logic.core.snippet import Snippet

snippet = Snippet(number_of_words_on_each_side=3)
doc = "The Shawshank Redemption is a high-rated movie directed by Frank Darabont and based on a story by Stephen King."
query = "Shawshank directed by Stephen King amir"
final_snippet, not_exist_words = snippet.find_snippet(doc, query)
print("Snippet:", final_snippet)
print("Not exist words:", not_exist_words)