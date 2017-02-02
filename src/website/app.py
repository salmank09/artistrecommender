from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from flask import Markup
import os

app = Flask(__name__)

# home page
@app.route('/', methods =['GET'])
def index():
    result = ''
    return render_template('index.html', result=Markup(result))

@app.route('/intro', methods =['GET'])
def intro():
    result = ''
    return render_template('post.html', result=Markup(result))



@app.route('/artistrecommender', methods =['GET', 'POST'])
def artistrecommender():
    artist_meta = pd.read_json('../../data/artist_meta.json')
    items_mat = np.array(list(artist_meta['formatted_features'].values)).astype(float)

    cos_sim = 1-pairwise_distances(items_mat, metric="cosine")

    cols = list(artist_meta.columns)

    similarity_columns = list(artist_meta['sc_alias'])

    cols.extend(similarity_columns)

    artist_aliases = list(artist_meta['sc_alias'])

    result = ''
    all_artists = ''

    sorted_artists = artist_meta.sort('artist_name')

    for index, row in sorted_artists.iterrows():
        try:
            all_artists += """<option value="{alias}">{artist}</option>""".format(alias = row['sc_alias'].decode('utf8', 'ignore').encode('utf-8')
                                                                              ,artist = row['artist_name'].decode('utf8', 'ignore').encode('utf-8'))
        except:
            print row['sc_alias'], row['artist_name']
            continue

    if request.method == 'POST':
        #read in alias provided by user
        alias = str(request.form['artist_alias'])

        index = artist_aliases.index(alias)

        alias_cossims = cos_sim[index]

        n = 30

        n_similar_aliases = np.asarray(artist_aliases)[alias_cossims.argsort()[-(n+1):][::-1][1:]]

        artist = artist_meta[artist_meta['sc_alias']==alias]['artist_name'].values[0]

        result += 'Here are the top recommended artists based on your search for <b>' + artist + '</b>:<br><br>'

        result += '<div class = "results_container">'

        for alias in n_similar_aliases:
            artist = artist_meta[artist_meta['sc_alias']==alias]['artist_name'].values[0]

            result += """
            <div class="col-xs-6 col-sm-3 col-md-2 col-lg-2">
              <div class="img-container">
                <a href='https://soundcloud.com/{alias}' target="_blank">
                    <img src="/static/images/{alias}.jpg" class="img-responsive" width="300px" height="300px" />
                </a>

                <div class="artist-name">
                    {artist}
                </div>
              </div>
            </div>
            """.format(alias=alias,
                       artist=artist)

    print len(all_artists)

    return render_template('artistrecommender.html', result=Markup(result), all_artists=Markup(all_artists))

@app.route('/about', methods =['GET'])
def about():
    result = ''
    return render_template('about.html', result=Markup(result))

@app.route('/predict', methods=['POST'])
def predict_results():

    return render_template('index.html', result=Markup(result))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded = True)
