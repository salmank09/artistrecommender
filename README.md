# musicrecommender

## Overview

This is a web-based application that makes artist recommendations to users based on
a queried artist. The application uses a content-based collaborative filtering model to determine
artist similarity measured by the amount of overlap in soundcloud followers for the queried artist and
the recommended artists; only recommending artists that have the most overlap.

Once a user searches for an artist and receives recommendations, the user has the ability to either follow
the recommended artists on Soundcloud and/or listen to some of their music via Spotify.

The artists in the application are all house musicians. The goal of this application is for users
to discover similar house musicians to those that they already enjoy.

However, this application can be scaled to encompass musicians across multiple genres.

## Data

The list of musicians for whom the application provides recommendations for was curated through
the following lists:

- Discogs.com first 25000 results for most requested house albums since 2010. Duplicate musicians and 'Various Artists' entries were removed.
https://www.discogs.com/search/?sort=have%2Cdesc&style_exact=House&genre_exact=Electronic&decade=2010&page=500
- Wikipedia's list of house musicians
https://en.wikipedia.org/wiki/List_of_house_music_artists

The list is additionally filtered down to musicians with Soundcloud and Spotify accounts with
500 followers or more; to reduce the likelihood of fake or incorrect accounts.

Followers data was scraped via Soundcloud's API.
