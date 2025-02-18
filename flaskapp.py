import json
from flask import Flask
from flask_cors import CORS
import csv


app = Flask(__name__)
CORS(app)


@app.route("/api/population/<city_name>", methods=["GET"])
def return_pop_by_region(city_name):
    response = {}
    try:
        # with 키워드는 파일 사용 후 자동으로 close
        with open("gender.csv", encoding='cp949') as file:
            data = csv.reader(file)

            for row in data:
                if f"{city_name}" in row[0]:
                    response["male"] = list(map(int, row[3:104]))
                    response["female"] = list(map(int, row[106:]))
                    break

        if len(response) == 0:
            response["error"] = "요청하신 도시에 대한 인구 데이터가 존재하지 않습니다"
    except Exception as ex:
            response["error"] = "내부 서버 오류"
    finally:
        return json.dumps(response)


if __name__ == '__main__':
    # app.run()
    print(json.dumps({
    1: (
        (["3", "5"], "8"),
        (["0", "0"], "0"),
        (["-1", "5"], "4"),
        (["100", "200"], "300"),
        (["-10", "-20"], "-30"),
        (["1", "-5"], "-4"),
        (["999999", "-123123"], "876876"),
        (["-100", "50"], "-50"),
        (["1000", "255"], "1255"),
        (["-990", "-550"], "-1540"),
    )}))
