from flask import Flask, render_template, request, jsonify
from elasticsearch import Elasticsearch
from flask_cors import CORS
from funk_svd.dataset import fetch_ml_ratings
from funk_svd import SVD
from sklearn.metrics import mean_absolute_error

import pandas as pd

app = Flask(__name__)
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def index():
	return {}

@app.route('/api/search', methods=['POST','GET'])
def search():
	body = request.get_json()
	query = body["query"]
	es = Elasticsearch(["http://localhost:9200/"], timeout=5000)
	result = es.search(index="listing_dataset",
					   doc_type="document",
					   body={
						   "query": {
								"match": {
									"text_content": query
								}
						   },
						   
					   })
	hits = result["hits"]["hits"]
	res = {"Search": []}
	for i in range(len(hits)):
		curr = {}
		doc_id = hits[i]["_source"]['docno']
		content = hits[i]["_source"]["text_content"].strip()
		neig = hits[i]["_source"]['neighborhood'].strip()
		lat = hits[i]["_source"]['latitude']
		longt = hits[i]["_source"]['longitude']
		pic = hits[i]["_source"]['pic_url']
		name = hits[i]["_source"]['name']
		curr["docno"] = doc_id
		curr["name"] = name
		curr["description"] = content
		curr["neighborhood"] = neig
		curr["latitude"] = lat
		curr["longitude"] = longt
		curr["pic_url"] = pic
		res["Search"].append(curr)
	return res


@app.route('/api/maps', methods=['POST'])
def plot_maps():
	body = request.get_json()
	neig = body['neighborhood']
	print(neig)
	df_s = pd.read_csv('./data/neighborhood_data.csv')
	cols = ['income_bin', 'white_bin', 'crime_bin']
	res = {}
	for col in cols:
		res[col] = str(df_s[df_s.community == neig][col].iloc[0])
	print(res)
	return res


@app.route('/api/recommend', methods=['POST','GET'])
def recommend():
	# body = request.get_json()

	df_rec = pd.read_csv('./data/rating.csv')
	#train
	train = df_rec.sample(frac=0.8, random_state=7)
	val = df_rec.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
	test = df_rec.drop(train.index.tolist()).drop(val.index.tolist()) 
	svd = SVD(lr=0.001, reg=0.005, n_epochs=100, n_factors=15,
		  early_stopping=True, shuffle=False, min_rating=-1, max_rating=1)
	svd.fit(X=train, X_val=val)
	pred = svd.predict(test)
	mae = mean_absolute_error(test['rating'], pred)
	print(f'Test MAE: {mae:.2f}')

	u_id = df_rec.sample().u_id.iloc[0]
	i_ids = list(df_rec.i_id.unique())
	fm = {'u_id': [u_id for i in range(len(i_ids))], 'i_id': i_ids}
	pred = pd.DataFrame.from_dict(fm)
	res = svd.predict(pred)
	pred['pred_rating'] = res
	pred.sort_values(by=['pred_rating'], ascending=False, inplace=True)
	N = 6
	df_1 = pred.head(N)

	df_2 = pd.read_csv('./data/listings_wrangled.csv')

	df = pd.merge(df_1, df_2, on='i_id')

	res = {"Recommend": []}
	for index, row in df.iterrows():
		curr = {}
		curr["docno"] = row['i_id']
		curr["name"] = row['name']
		curr["description"] = row['description']
		curr["neighborhood"] = row['neighbourhood_cleansed']
		curr["latitude"] = row['latitude']
		curr["longitude"] = row['longitude']
		curr["pic_url"] = row['picture_url']
		res["Recommend"].append(curr)
	
	return res

