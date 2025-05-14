import mido
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_midi():
    """
    MIDI visualizer that creates a piano roll view with ALL notes labeled,
    including bass clef notes.
    """
    # File path - change this to your MIDI file
    midi_file = "./samples/output_pitch.mid"
    
    # Load the MIDI file
    mid = mido.MidiFile(midi_file)
    
    print(f"Visualizing MIDI file: {midi_file}")
    print(f"Number of tracks: {len(mid.tracks)}")
    
    # Determine the maximum time
    max_ticks = 0
    for track in mid.tracks:
        current_ticks = 0
        for msg in track:
            current_ticks += msg.time
            if current_ticks > max_ticks:
                max_ticks = current_ticks
    
    # Convert ticks to seconds (approximate, assuming 120 BPM if not specified)
    tempo = 500000  # Default tempo (500000 microseconds per beat = 120 BPM)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
    
    # Calculate seconds per tick
    seconds_per_tick = tempo / 1000000 / mid.ticks_per_beat
    max_time = max_ticks * seconds_per_tick
    
    print(f"MIDI file duration: {max_time:.2f} seconds")
    
    # Define note name mapping
    note_names = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }
    
    # Function to get the note name with octave (e.g., A4, C5)
    def get_note_name(midi_note):
        octave = midi_note // 12 - 1  # MIDI note 60 is C4
        note = midi_note % 12
        return f"{note_names[note]}{octave}"
    
    # Create a figure with two subplots: 
    # 1. Piano roll
    # 2. Legend for tracks
    fig, (ax_roll, ax_legend) = plt.subplots(2, 1, figsize=(14, 9), 
                                             gridspec_kw={'height_ratios': [4, 1]})
    
    # Track colors
    track_colors = plt.cm.tab10.colors  # Use 10 colors for tracks
    
    # Keep track of notes to define y-axis limits
    all_notes = []
    legend_handles = []
    
    # Dictionary to store note rectangles for later labeling
    note_rectangles = []
    
    # Process each track
    for track_idx, track in enumerate(mid.tracks):
        # Get a color for this track
        track_color = track_colors[track_idx % len(track_colors)]
        
        # Get track name
        track_name = f"Track {track_idx+1}"
        for msg in track:
            if msg.type == 'track_name':
                track_name = msg.name
                break
        
        # Add to legend
        legend_handles.append(plt.Line2D([0], [0], color=track_color, lw=4, label=track_name))
        
        # Process messages in this track
        current_time = 0
        active_notes = {}  # note -> (start_time, velocity)
        
        for msg in track:
            # Update time
            current_time += msg.time
            
            # Process note messages
            if msg.type == 'note_on' and msg.velocity > 0:
                # Note start
                active_notes[msg.note] = (current_time, msg.velocity)
                all_notes.append(msg.note)
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Note end
                if msg.note in active_notes:
                    start_time, velocity = active_notes[msg.note]
                    
                    # Convert to seconds
                    start_sec = start_time * seconds_per_tick
                    end_sec = current_time * seconds_per_tick
                    
                    # Draw the note rectangle on the piano roll
                    rect = plt.Rectangle(
                        (start_sec, msg.note - 0.4), 
                        end_sec - start_sec, 
                        0.8, 
                        color=track_color,
                        alpha=0.7
                    )
                    ax_roll.add_patch(rect)
                    
                    # Store the note information for later labeling
                    note_rectangles.append({
                        'start': start_sec,
                        'end': end_sec,
                        'note': msg.note,
                        'note_name': get_note_name(msg.note),
                        'rect': rect
                    })
                    
                    # Remove from active notes
                    del active_notes[msg.note]
    
    # Calculate y-axis range
    if all_notes:
        min_note = max(0, min(all_notes) - 5)
        max_note = min(127, max(all_notes) + 5)
    else:
        min_note = 60 - 24  # Default range around middle C, extended lower for bass clef
        max_note = 60 + 24
    
    # Now add labels to all notes
    for note_info in note_rectangles:
        # Always add the note name
        note_center_x = note_info['start'] + (note_info['end'] - note_info['start']) / 2
        note_center_y = note_info['note']
        
        # Determine text color (white for very dark rectangles)
        rect_color = note_info['rect'].get_facecolor()
        text_color = 'white' if np.mean(rect_color[:3]) < 0.5 else 'black'
        
        # Add the note name text
        ax_roll.text(
            note_center_x,
            note_center_y,
            note_info['note_name'],
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold',
            color=text_color
        )
    
    # Add legend to the bottom plot
    ax_legend.legend(handles=legend_handles, loc='center', ncol=min(len(legend_handles), 4))
    ax_legend.axis('off')  # Hide axes for legend subplot
    
    # Set up the piano roll plot
    ax_roll.set_xlim(0, max_time)
    ax_roll.set_ylim(min_note - 0.5, max_note + 0.5)
    
    # Add piano keyboard layout on the left
    key_width = 0.8
    for i in range(min_note, max_note + 1):
        # Determine if it's a black key
        is_black = (i % 12) in [1, 3, 6, 8, 10]  # C#, D#, F#, G#, A#
        color = 'black' if is_black else 'white'
        edge_color = 'white' if is_black else 'black'
        
        # Draw the key
        piano_key = plt.Rectangle(
            (-key_width, i - 0.4), 
            key_width, 
            0.8, 
            color=color,
            edgecolor=edge_color,
            zorder=2
        )
        ax_roll.add_patch(piano_key)
        
        # Add note name for each key
        note_name = get_note_name(i)
        ax_roll.text(-key_width/2, i, note_name, ha='center', va='center', 
                   fontsize=8, color='white' if is_black else 'black')
    
    # Custom Y-axis tick labels showing note names
    ytick_positions = []
    ytick_labels = []
    
    # Add tick marks for all C and F notes within our range for better orientation
    for note in range(min_note, max_note + 1):
        if note % 12 == 0 or note % 12 == 5:  # C or F notes
            ytick_positions.append(note)
            ytick_labels.append(get_note_name(note))
    
    ax_roll.set_yticks(ytick_positions)
    ax_roll.set_yticklabels(ytick_labels)
    
    # Add grid lines for measures (assuming 4/4 time signature)
    beats_per_measure = 4
    seconds_per_measure = beats_per_measure * tempo / 1000000
    for i in range(int(max_time / seconds_per_measure) + 1):
        ax_roll.axvline(x=i * seconds_per_measure, color='gray', linestyle='--', alpha=0.5)
    
    # Also add subtle grid lines for all semitones
    for note in range(min_note, max_note + 1):
        ax_roll.axhline(y=note, color='gray', linestyle='-', alpha=0.1)
    
    # Set labels
    ax_roll.set_title(f'Piano Roll View: {os.path.basename(midi_file)} with Note Names')
    ax_roll.set_xlabel('Time (seconds)')
    ax_roll.set_ylabel('Note')
    
    # Tight layout
    plt.tight_layout()
    
    # Save and show the visualization
    output_file = os.path.splitext(midi_file)[0] + '_piano_roll_all_labeled.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    plt.show()

# Simple main function with no arguments
def main():
    visualize_midi()

if __name__ == "__main__":
    main()