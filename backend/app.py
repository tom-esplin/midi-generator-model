from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)

    notes = data.get("notes", [])
    tempo = data.get("tempo", 120)
    genre = data.get("genre", "jazz")
    length_measures = data.get("lengthMeasures", 4)
    start_measure = data.get("startMeasure", None)

    if not notes:
        return jsonify({"error": "No notes provided"}), 400

    beat_duration = 60.0 / tempo
    measure_duration = beat_duration * 4

    # Where to place generated notes (seconds)
    if start_measure is not None:
        gen_start = (start_measure - 1) * measure_duration
    else:
        gen_start = max(n["startTime"] + n["duration"] for n in notes)

    gen_length_sec = length_measures * measure_duration

    # Phase 1: duplicate input notes, scaled to fit within the requested
    # length and shifted to the start position.
    input_start = min(n["startTime"] for n in notes)
    input_end = max(n["startTime"] + n["duration"] for n in notes)
    input_span = max(input_end - input_start, 0.001)

    generated = []
    for n in notes:
        relative = (n["startTime"] - input_start) / input_span
        new_start = gen_start + relative * gen_length_sec
        scale = gen_length_sec / input_span
        new_dur = n["duration"] * scale

        generated.append(
            {
                "pitch": n["pitch"],
                "velocity": n["velocity"],
                "startTime": round(new_start, 6),
                "duration": round(max(new_dur, beat_duration / 4), 6),
            }
        )

    return jsonify({"notes": generated})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
