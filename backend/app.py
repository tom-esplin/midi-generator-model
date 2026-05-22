import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.generation import VALID_GENRES, generate_continuation

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

    genre_key = genre.lower().strip()
    if genre_key not in VALID_GENRES:
        return jsonify({
            "error": f"Unsupported genre '{genre}'. "
                     f"Choose from: {', '.join(sorted(VALID_GENRES))}"
        }), 400

    try:
        generated = generate_continuation(
            notes,
            tempo=float(tempo),
            genre=genre_key,
            length_measures=int(length_measures),
            start_measure=int(start_measure) if start_measure is not None else None,
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Generation failed: {exc}"}), 500

    return jsonify({"notes": generated})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
