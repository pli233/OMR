from music21 import converter

score = converter.parse("3.mid")
score.write('lily.pdf', fp='output.ly')