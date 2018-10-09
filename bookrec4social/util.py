import requests
from xml.etree import ElementTree
import os
import sys
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
from numpy import linalg as LA

currentpath = str(os.path.dirname(os.path.abspath(__file__)))

not_found_error_message = "That username doesn't seem to exist on Goodreads, I'm sorry"
private_error_message = "This user account is private, I'm sorry"
no_ratings_error_message = "You don't have any ratings on the books we have access to, I'm sorry"



def get_id_from_username(username, api_key):
    response = requests.get('https://www.goodreads.com/user/show/?key=' + api_key + '&username=' + username + '&format=xml')
    tree = ElementTree.fromstring(response.content)
    try:
        user_id = tree.find('user').find('id').text
        return user_id
    except:
        # raise ValueError('Invalid Goodreads username, not id returned')
        return None

def get_url_from_id(user_id, api_key):
    response = requests.get('https://www.goodreads.com/user/show/?key=' + api_key + '&id=' + user_id + '&format=xml')
    tree = ElementTree.fromstring(response.content)
    try:
        user_url = tree.find('user').find('link').text
        return user_url
    except:
        # raise ValueError('Invalid Goodreads username, not id returned')
        return None


    
def get_user_vector(user_input, mapper):
    """ Gets the user ratings vector of a user

    Args:
        user_input::str
            username of the user
        mapper::dict
            maps the goodreads book id to our ids

    Returns:
        user_vector::np.array
            an array of 10000 ratings for the given user
        error_message::str
            an error message string, if there is an error
    """
    try:
        sparse_q = scipy.sparse.load_npz(currentpath+'/static/data/cached_users/user_' + user_input + '.npz')
        q = sparse_q.toarray()
        q = np.array(q[0].tolist())
        print('found user_vector...')
        return q, None
    except:
        q = np.zeros((10000), dtype = np.float)
        # api_key = secret.API_KEY
        api_key = '4nbvcBRfYZ1MSL06ARuw'
        if not user_input.isdigit():
            user_id = get_id_from_username(user_input, api_key)
        else:
            user_id = user_input

        if user_id is None:
            return None, not_found_error_message

        page = 1
        total_valid_reviews = 0
        while True:
            response = requests.get('https://www.goodreads.com/review/list/?v=2&id=' + user_id + '&shelf=read&format=xml&key=' + api_key + '&per_page=200&page=' + str(page))
            tree = ElementTree.fromstring(response.content)
            reviews = tree.find('reviews')
            if reviews is None:
                return None, private_error_message
            for review in reviews:
                goodreads_book_id = str(review.find('book').find('id').text)
                if goodreads_book_id in mapper:
                    book_id = int(mapper[goodreads_book_id])
                    rating = int(review.find('rating').text)
                    q[book_id - 1] = float(rating)
                    total_valid_reviews += 1
            page += 1

            print(len(reviews))
            if len(reviews) < 1:
                break

        print("total valid reviews: %s" % (total_valid_reviews))
        if total_valid_reviews < 1:
            return None, no_ratings_error_message

        q = feature_scaling(q)

        # Disable this until we find a 'smart' caching solution
        print('saving user_vector...')
        scipy.sparse.save_npz(currentpath+'/static/data/cached_users/user_'+user_input+'.npz', scipy.sparse.csr_matrix(q))

        return q, None


def get_friends_information(user_input, q, mapper):

    from lxml import html
    password = '320501xxy'
    session_requests = requests.session()

    login_url = 'https://www.goodreads.com/'
    result  = session_requests.get(login_url)
    tree = html.fromstring(result.text)
    authenticity_token = list(set(tree.xpath("//input[@name='authenticity_token']/@value")))[0]
    print(authenticity_token)
    payload = {
        "user[email]": "xyx1811@gmail.com",
        "user[password]": password,
        "authenticity_token" : authenticity_token
        }        
    result = session_requests.post(
        login_url,
        data = payload,
        headers = dict(referer=login_url)
        )
    
    users = []

    api_key = '4nbvcBRfYZ1MSL06ARuw'
    url = get_url_from_id(user_input, api_key)
    print(url)
    result = session_requests.get(url, headers = dict(referer = url))
    tree = html.fromstring(result.content)
    friend_names=tree.xpath("//div[@class='friendName']/a/text()")
    friend_ids=tree.xpath("//div[@class='friendName']/a/@href")
    friend_images=tree.xpath("/div.bigBoxBody/img/@src")
    friend_ids=[ ''.join(list(filter(str.isdigit, item))) for item in friend_ids]
    print(friend_names)
    print(friend_ids)
    friend_class = {}
    
    friend_vec = np.zeros((10000), dtype = np.float)
    ncount = 0
    if len(friend_ids) == len(friend_names):
        for ix in range(len(friend_ids)):
            vec, err = get_user_vector(friend_ids[ix], mapper)
            # print(err)
            if err!=200 and err!=None:
                # print(err)
                continue
            
            ncount = ncount + 1
            score = np.dot(vec, q)
            # print(LA.norm(vec))
            # score = score/(LA.norm(vec)*LA.norm(q))
            print("score: %.2e"%(score))
            print("#books: %d"%(len(np.nonzero(vec)[0])))
            # print(score)
            if score > 0:
                friend_vec = np.add(vec, friend_vec)
            friend_class[friend_ids[ix]] = {'ID':friend_ids[ix], 'Name':friend_names[ix], 'Score':score, 'Books':len(np.nonzero(vec)[0])}
    print(friend_class)
    
    return friend_names, friend_class, friend_vec, ncount



def feature_scaling(q):
    """ Scales the user features using the mean and the
    standard deviation.
    """
    if q.dtype != np.float:
        q = q.astype(np.float)
    nonzero = np.nonzero(q)
    nonzero_ratings = q[nonzero]
    mean = np.mean(nonzero_ratings)
    std = np.std(nonzero_ratings)
    print('Mean: %s' % (mean))
    print('S.D: %s' % (std))
    if std == 0:
        q[nonzero] = (1.0 + q[nonzero] - mean)
    else:
        q[nonzero] = (1.0 + q[nonzero] - mean) / std
    return q


'''
Recommender functions
'''


def chunker(top_books):
    # chunk into groups of 3 to display better in web app
    chunks = []
    current_chunk = []
    for i in range(len(top_books)):
        if len(current_chunk) < 3:
            current_chunk.append(top_books[i])
        else:
            chunks.append(current_chunk)
            current_chunk = [top_books[i]]

    chunks.append(current_chunk)
    return chunks


def get_books_from_indices(top_book_indices, books):
    top_books = []
    for i in range(len(top_book_indices)):
        book_id = top_book_indices[i]
        book = books.iloc[book_id - 1]  # index is book_id - 1
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)
    return top_books


def get_top_n_recs(result, books, n, q):
    recs = []
    for i in range(len(result)):
        if q[i] == 0:  # book user hasn't already rated
            recs.append((i, result[i]))
        else:
            recs.append((i, float('-inf')))
    recs = sorted(recs, key=lambda tup: tup[1], reverse=True)

    top_books = []
    for i in range(n):
        book_id = recs[i][0]
        book = books.iloc[book_id]
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)

    return top_books


def most_popular(books, n):
    top_books = []
    for i in range(n):
        book = books.iloc[i]
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)

    return top_books


def map_user(q, V):
    # map new user to concept space by q*V
    user_to_concept = np.matmul(q, V)
    # map user back to itme space with user_to_concept * VT
    result = np.matmul(user_to_concept, V.T)
    return result
