from Logic.core.preprocess.preprocess import Preprocessor


def run_test_preprocess():
    pp = Preprocessor(
        [
         {
          "id": 1,
          "stars": ["Amirhossein Akbari", "Akbar Amiri", "henry"],
          "genres": ["Drama", "Adventure"],
          "summaries": ["Amirhossein is good at do nothing!", "this, only this one is fake one."]
         },
         {
          "id": 2,
          "stars": ["Dr. Mahdieh Soleimani", "Soli Dr. Mahdieh"],
          "genres": ["Action", "Crime"],
          "summaries": ["she want to kill us! help!", "any body hear me?"]
         }
        ], "../core/preprocess/stopwords.txt")
    print(pp.preprocess())