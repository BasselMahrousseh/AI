import pretty_midi
# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()
# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=7)
# Iterate over note names, which will be converted to note number later
i=100
j=0
for note_name in ['G5','A4','C6','B6']:
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance, starting at 0s and ending at .5s
    note = pretty_midi.Note(
        velocity=50, pitch=note_number, start=j, end=j+0.5)
    j=j+0.5
    # Add it to our cello instrument
    cello.notes.append(note)
    i+=100
# Add the cello instrument to the PrettyMIDI object
cello_c_chord.instruments.append(1)
# Write out the MIDI data
cello_c_chord.write('cello-C-chord.mid')
