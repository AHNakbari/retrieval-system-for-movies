from Logic.core.spell_correction import SpellCorrection, create_all_documents

spell_correction = SpellCorrection(create_all_documents())
query = "Provied a lst of moveis alon with thier summries, geners," \
        " and the starrs of each movei. Additinally," \
        " includd any relevent infrmation about the directrs or notable acolades asociated with the flms."
print(spell_correction.spell_check(query, True))

