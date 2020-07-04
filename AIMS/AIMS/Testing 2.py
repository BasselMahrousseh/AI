from mido import MidiFile
from visual_midi import Plotter
from pretty_midi import PrettyMIDI

mid = MidiFile('cello-C-chord.mid', clip=True)
#mid = MidiFile('mashup.mid', clip=True)
print(mid)
print('######################################')
for track in mid.tracks:
    print(track)
print('######################################')
for msg in mid.tracks[0]:
    print(msg)



#pm = PrettyMIDI("cello-C-chord.mid")
#plotter = Plotter()
#plotter.show(pm, "example-01.html")