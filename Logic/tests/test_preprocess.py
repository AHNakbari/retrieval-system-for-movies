from Logic.core.preprocess import Preprocessor
print("start the preprocess test")
pp = Preprocessor(
    ["Being one of the most anticipated movies of the year, this, that!",
     "'The Great Adventure' did not disappoint. Critics say that it's a masterpiece that redefines the genre.",
     "Visit www.thegreatadventure.com for exclusive behind-the-scenes content! Why wait? Check it out now.",
     "Email us at info@thegreatadventure.com for more details.",
     "This film, directed by Jane Doe, showcases the journey of a lifetime,",
     "being both thrilling and emotionally captivating.",
     "Should you watch it? Absolutely, without a doubt!"], "../core/stopwords.txt")
print(pp.preprocess())