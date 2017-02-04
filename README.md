# artistrecommender

## Overview

The ArtistRecommender application is a web-based music recommender system which recommends house musicians. The unique aspect of this recommender system is that it recommends musicians to listen to, as opposed to individual songs to listen to. I find it valuable to be recommended artists that I might like because if I do happen to like a song by the artist, there is a fair chance that I would like and listen to additional songs by them; therefore having a more immersive listening experience than listening to multiple songs by a variety of artists back to back.

The application uses a content-based collaborative filtering model to determine artist similarity learned by the amount of overlap in Soundcloud followers for the queried artist and the recommended artists; only recommending artists that are most similar to each other.

Once a user searches for an artist and receives recommendations, the user has the ability to either follow the recommended artists on Soundcloud.

The application can be found at https://salstuff.com/artistrecommender

For further details on how the application was developed, continue reading.

## Data Gathering

1. The list of musicians for whom the application provides recommendations for was gathered through the following lists:

- Discogs.com first 25000 results for most requested house albums since 2010. Duplicate musicians and 'Various Artists' entries were removed.
https://www.discogs.com/search/?sort=have%2Cdesc&style_exact=House&genre_exact=Electronic&decade=2010&page=500

- Wikipedia's list of house musicians
https://en.wikipedia.org/wiki/List_of_house_music_artists

Duplicate names were removed.

2. Each of these musicians' Soundcloud pages were queried for via Google - those of whom that had a Soundcloud profile appearing in the first search result were kept, the rest were discarded and assumed to not have Soundcloud pages.

Some artists, including those that were not closely named, linked to the same Soundcloud page. Duplicates were removed by keeping the artist that met one or more of the following conditions lending credit to its legitimacy:

A. The artist name more closely matches the name of the Soundcloud page than other artists
B. The artist appeared higher in the original list if it came from the Discogs.com list
C. Gut feeling ;)

3. In attempt of further legitimacy validation, of the remaining artists, additional artists were removed for not meeting at least one of the following conditions that are common for popular and legitimate artist pages:

A. Follower count was less than 500
B. Soundcloud account subscription was not a Pro or Pro Plus account. i.e. it was a Free account

The final list of artists amounted to 5,625.

4. For each of the remaining 5,625 artists - their followers were stored in MongoDB. Currently the model contains approximately 4,800 artists and their followers, who were scraped in order of their number of followers in ascending order.

Followers for the remaining artists are currently being scraped and are not expected to finish for several months, possibly June 2017. I intend to update the application on a periodic basis as the data for more popular artists becomes available.

## Data Exploration

- The dataset currently consists of 28M artist-follower combinations, among whom 6M followers were unique.

- 26,000 followers were following more than 100 of the 4,800 artists

- 500,000 followers were following more than 10 of the 4,800 artists

If we define an active user to be following 10 or more of the artists in our dataset, only about 1% of the dataset contains active users, giving us a fairly sparse matrix.

These numbers would have to be compared to similar metrics against an industry standard to have some context on the quality of our dataset.

## Model Development

I took a collaborative filtering approach to making recommendations leveraging matrix factorization algorithms which have historically offered best in class performance relative to other models. The matrix factorization algorithm used on this dataset is the Alternating Least Squares (ALS) algorithm.

Within the context of this dataset, the model creates a sparse matrix of Users x Artists (UA) with the row, column indices containing 0's and 1's - acting as booleans for whether or not a user is following an artist. From there, the model generates two additional matrices (feature matrices) whose dot product approximates the original UA matrix. The ALS algorithm allows for parallelization when generating the two feature matrices, and this parallelization is leveraged by Spark for optimization.

The approximation is intended to minimize the root mean squared error between the original matrix and the approximation and act as an indicator of whether or not users would follow an artist - which is particularly useful for predicting the likelihood of relationships between users and artists when they are not already related.

The model itself was developed through grid-searching hyper-parameters available in Spark MLLib's Alternating Least Squares algorithm including rank, the number of latent features to "discover", and the learning parameter, lambda, with rmse as the error metric.

The memory on my local machine was limited and not powerful enough to train the model, requiring me to resort to training on an Elastic Map Reduce cluster hosted on Amazon Web Services.

Although I did not take this approach, a potentially better approach of developing the model would be to minimize the error between the values corresponding to a user's top n recommended artists, and the actual values for those artists for that user, for each user. This could be considered a better approach because we are only evaluating the effectiveness of our approximation on recommendations that we care about, i.e. the ones the user sees.

## Recommendation

One of the two matrices generated from the approximation of UA is an Artist x Features matrix. The feature weights are scores "discovered" by the ALS algorithm about the artists for each of the feature categories.

I precomputed the cosine similarities for each artist relative to the rest of the artists on their feature vectors. Given this matrix, I take the queried artist and return the n most similar artists to that artist by their cosine similarity scores, 1 being very similar, 0 being not similar at all.

Precomputing the cosine similarities allows the application to do an easy lookup job rather than an expensive compute job which greatly expedites the return of the query results.

## Evaluation

Recommender systems are traditionally difficult to evaluate, as whether a recommendation is good or not can be largely subjective.

To come up with a measurement of evaluation - I intend to assess whether the model is working at its most fundamental level. i.e. assess whether it is recommending artists who have the most followers in common with the queried artist. To do this I will take a test list of artists, for each of them determining the percentage of overlap between the recommended artists and the artists they have the most followers in common with.

I also intend to sort the artists grouped by each of the ten features in descending order of their feature weights to determine which artists dominate the individual features most - from there I can attempt to determine what is similar about these artists.

## Deployment

The application is deployed using Flask and is hosted on an EC2 cluster via Amazon Web Services.

## FAQ
Q: Why did you choose only the House genre?

A: I chose to focus on house musicians in particular because that is the genre of music I enjoy listening to most at this time and am interested in discovering new house musicians to listen to.

Q: I listen to house music but don't recognize many of the artists in your application. Why don't I see more popular house musicians to choose from?

A: Scraping followers via Soundcloud's API is time consuming. It can take several minutes to scrape just 1000 followers. In order to get an MVP going I decided to build the model after a week of scraping followers. At this point, I had scraped followers for approximately 4,800 of the 5,625 artists being scraped, which are being scraped in ascending order of number of followers.

Scraping for followers of the remaining artists is not expected to complete for several more months, as some of these artists have close to 10M followers. I will periodically update the model to include more popular artists to base recommendations off of.

Q: Is your process scalable? Can I use it for other genres?

A: Yes. Data can be gathered for additional musicians across additional genres and a recommender model trained using the same approach.

Q: Why didn't you consider collaborative filtering algorithms other than Alternating Least Squares to train your model?

A: The ALS algorithm is the only collaborative filtering algorithm natively supported by Spark MLlib. I wanted to get experience using Spark, and given the size of the dataset (28M rows) it also made the most sense to use.
