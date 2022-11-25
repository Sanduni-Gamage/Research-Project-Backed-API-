from email import message
import json
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import Flask, Response, request
from inference import *

app = Flask(__name__) 
CORS(app)

@app.route("/herbal", methods=["POST"])
def herbal():
    try:
        HerbalImagefile = request.files['herbal']
        filename = secure_filename(HerbalImagefile.filename)
        filepath = os.path.join('herbals', filename)
        HerbalImagefile.save(filepath)

        herbal = predict_cnn(filepath)
        diseases, treatments = predict_automl(herbal)
        if (diseases != []) and (treatments != []) :
            return Response(
                        json.dumps({
                                "herbal" : f"{herbal}", 
                                "diseases" : f"{diseases}", 
                                "treatments" : f"{treatments}"
                                }), 
                                mimetype='application/json'
                            )
        else:
            return Response(
                response=json.dumps({
                    "status": "Unscuccessful",
                    "error": "Herbs,Diseases and Treatments not found in data base"
                }),
                status=200,
                mimetype="application/json"
            )        


    except Exception as e:
        return Response(
            response=json.dumps({
                "status": "Unscuccessful",
                "error": f"{e}"
            }),
            status=500,
            mimetype="application/json"
        )

@app.route("/distribution", methods=["GET"])
def distribution():
    try:
        create_distribution_visualizations()
        return Response(
                    json.dumps({
                            "folium_map_path" : f"{folium_map_path}", 
                            "pie_chart_path" : f"{pie_chart_path}"
                            }), 
                            mimetype='application/json'
                        )


    except Exception as e:
        return Response(
                    response=json.dumps({
                        "status": "Unscuccessful",
                        "error": str(e)
                    }),
                    status=500,
                    mimetype="application/json"
        )

@app.route("/recommndation", methods=["POST"])
def recommndation():
    try:
        message = request.json
        userid = message['userid']
        postid = int(message['postid'])

        m1_json, m2_json = run_recommendation(userid, postid)
        return Response(
                    json.dumps({
                            "m1_json" : f"{m1_json}", 
                            "m2_json" : f"{m2_json}"
                            }), 
                            mimetype='application/json'
                        )


    except Exception as e:
        return Response(
                    response=json.dumps({
                        "status": "Unscuccessful",
                        "error": str(e)
                    }),
                    status=500,
                    mimetype="application/json"
        )


if __name__ == '__main__':
    app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000
            )