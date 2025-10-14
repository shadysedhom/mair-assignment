import datetime
import time
import os

class DialogueLogger:
    """A class to handle logging of dialogue turns and saving transcripts."""
    def __init__(self):
        self.transcript = []
        self.turn_count = 0
        self.system_turns = 0
        self.user_turns = 0
        self.start_time = time.time()

    def log_turn(self, speaker, utterance):
        """Logs a single turn of the dialogue."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.turn_count += 1
        if speaker == "System":
            self.system_turns += 1
        else:
            self.user_turns += 1

        log_entry = (
            f"Timestamp: {timestamp}\n"
            f"Turn: {self.turn_count}\n"
            f"Speaker: {speaker}\n"
            f"Utterance: {utterance}\n"
        )
        self.transcript.append(log_entry)

    def save(self):
        """Saves the complete dialogue transcript to a unique file."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Calculate MM:SS format
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_mm_ss = f"{minutes:02d}:{seconds:02d}"

        session_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join("saved_transcripts", f"dialogue_{session_timestamp}.txt")

        with open(filename, 'w') as f:
            f.write("--- Dialogue Transcript ---\n\n")
            f.write("\n".join(self.transcript))
            f.write(f"\n--- End of Dialogue ---\n")
            f.write(f"Total Duration (MM:SS format): {duration_mm_ss}\n")
            f.write(f"Total Duration (in seconds): {duration:.2f} seconds\n")
            f.write(f"Total Turns: {self.turn_count}\n")
            f.write(f"System Turns: {self.system_turns}\n")
            f.write(f"User Turns: {self.user_turns}\n")
        
        print(f"[Dialogue transcript saved to {filename}]")
