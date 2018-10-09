from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import logging
import pickle
import csv
import os
from bookrec4social import app

# Custom libraries

from bookrec4social.util import get_user_vector, chunker, get_top_n_recs, map_user, most_popular, get_books_from_indices, not_found_error_message, get_friends_information


''' GLOBALS
'''
bookid_to_title = None
title_to_bookid = None
mapper_id = None
item_matrix = None
qi = None  # item to concept matrix from SVD
top_recs_each_book_item_matrix = None
top_recs_each_book_feature_matrix = None
books = None
titles = None
currentpath = str(os.path.dirname(os.path.abspath(__file__)))

''' DATA LOADING FUNCTIONS
'''


def load_books():
    """ Loads in the books and titles from the pickled dataframe """
    """['author', 'description', 'image_url', 'title', 'url']"""
    global books, titles
    if books is None or titles is None:
        titles = []
        books = pd.read_pickle(currentpath+'/static/data/books_dataframe')
        for index, row in books.iterrows():
            titles.append(row['title'])
        titles.sort()
        print('books loaded')


def load_title_mappers():
    """ Loads in the title mappers using books.csv """
    global bookid_to_title, title_to_bookid
    if bookid_to_title is None or title_to_bookid is None:
        bookid_to_title = {}
        title_to_bookid = {}
        filename = currentpath+'/static/data/books.csv'
        with open(filename, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                bookid_to_title[line[0]] = line[10]
                title_to_bookid[line[10]] = line[0]
        print('books mapper loaded')


def load_id_mapper():
    """ Loads in the id mapper using books.csv.

    This maps goodreads book ids to our ids.
    """
    global mapper_id
    if mapper_id is None:
        mapper_id = {}
        filename = currentpath+'/static/data/books.csv'
        with open(filename, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                mapper_id[line[1]] = line[0]
        print('mapper_id loaded')


def load_item_matrix():
    """ Loads in the item to concept matrix """
    global item_matrix
    if item_matrix is None:
        item_matrix = np.load(currentpath+'/static/data/item_matrix.npy')
        print('item matrix loaded')


def load_top_recs_each_book():
    global top_recs_each_book_item_matrix, top_recs_each_book_feature_matrix
    if top_recs_each_book_feature_matrix is None or top_recs_each_book_item_matrix is None:
        f = open(currentpath+'/static/data/top_recs_each_book_item_matrix.pkl', "rb")
        top_recs_each_book_item_matrix = pickle.load(f)
        f.close()

        f = open(currentpath+'/static/data/top_recs_each_book_feature_matrix.pkl', "rb")
        top_recs_each_book_feature_matrix = pickle.load(f)
        f.close()
        print('top recs for each book loaded')


def load_svd_matrix():
    global qi
    if qi is None:
        qi = np.load(currentpath+'/static/data/svd.npy')
        print('svd matrix loaded')


def load_data():
    global titles
    load_title_mappers()
    load_id_mapper()
    # load_svd_matrix()
    load_books()
    load_item_matrix()
    # load_top_recs_each_book()
    return render_template('book_list.html', titles=titles)


''' HOME PAGE
'''


# @app.route('/')
# def home():
#     return render_template('start.html')


# @app.route('/', methods=['POST'])
# def home_post():
#     if 'load' in request.form:
#         return redirect(url_for('recommender'))


''' SETUP PAGE
'''


# @app.route('/setup')
# def setup():
#     return render_template('setup.html')


# @app.route('/setup', methods=['POST'])
# def setup_post():
#     if 'load' in request.form:
#         return redirect(url_for('recommender'))


''' RECOMMENDER PAGE
'''


@app.route('/')
def home():
    return load_data()


@app.route('/', methods=['POST'])
def home_post():
    global item_matrix, books, title_to_bookid, cosine_sim_item_matrix, cosine_sim_feature_matrix
    # if 'book_recs' in request.form:
    #     text = request.form['books']

    #     if text not in title_to_bookid:
    #         return render_template('book_list.html',
    #                                response=not_found_error_message,
    #                                titles=titles)

    #     book_id = int(title_to_bookid[text])
    #     top_book_indices = top_recs_each_book_item_matrix[book_id]
    #     top_books = get_books_from_indices(top_book_indices, books)
    #     chunks = chunker(top_books)
    #     return render_template('book_list.html',
    #                            toPass=chunks,
    #                            titles=titles,
    #                            response='Showing Recommendations for: ' + text)

    # if 'book_similar' in request.form:
    #     text = request.form['books']

    #     if text not in title_to_bookid:
    #         return render_template('book_list.html',
    #                                response=not_found_error_message,
    #                                titles=titles)

    #     book_id = int(title_to_bookid[text])
    #     top_book_indices = top_recs_each_book_feature_matrix[book_id]
    #     top_books = get_books_from_indices(top_book_indices, books)
    #     chunks = chunker(top_books)
    #     return render_template('book_list.html',
    #                            toPass=chunks,
    #                            titles=titles,
    #                            response='Showing Similar Books to: ' + text)

    if 'user_recs' in request.form:
        text = request.form['text']

        q, error_message = get_user_vector(text, mapper_id)
        if error_message:
            return render_template('book_list.html',
                                   response=error_message,
                                   titles=titles)

        # Get recs using item_matrix
        top_books = get_top_n_recs(map_user(q, item_matrix), books, 60, q)
        chunks = chunker(top_books)

        # Get recs using svd
        # top_books = get_top_n_recs(map_user(q, qi), books, 99, q)

        return render_template('book_list.html',
                               toPass=chunks,
                               titles=titles,
                               # response='Showing Recommendations for: ' + text)
                               response='')

    if 'user_friends' in request.form:
        text = request.form['text']
        q, error_message = get_user_vector(text, mapper_id)
        
        if error_message:
            return render_template('book_list.html',
                                   response=error_message,
                                   titles=titles)

        # Get recs using item_matrix
        friend_name, friend_class, friend_vec, ncount = get_friends_information(text, q, mapper_id)
        print(ncount)
        # print(np.sum(friend_vec))
        # print(np.sum(q))
        
        if ncount == 0:
            q_new = np.add(q, 0)
            return render_template('book_list.html',
                               toPass=chunks,
                               titles=titles,
                               # response='Showing Recommendations for: ' + text)
                                response='Not enough friend data collected',
                                friends=friend_name )
            
        else:
            q_new = np.add(q, friend_vec)
            top_books = get_top_n_recs(map_user(q_new, item_matrix), books, 60, q)
            chunks = chunker(top_books)
            return render_template('book_list.html',
                                       toPass=chunks,
                                       titles=titles,
                                       response='Showing Recommendations for: ' + text + ' & friends')
        

    # if 'most_popular' in request.form:
    #     top_books = most_popular(books, 99)
    #     chunks = chunker(top_books)
    #     return render_template('book_list.html',
    #                            toPass=chunks,
    #                            titles=titles,
    #                            response='99 Most Popular Books')
    else:
        return 'ERROR'


