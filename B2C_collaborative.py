# Import Libraries
import numpy as np
import pandas as pd
import mysql.connector
import time
import datetime
from collections import Counter
from mysql.connector import errorcode
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


CURRENT_YEAR = datetime.datetime.now().year


def _db_conn(customer_query, database='getabstract'):
    """ This function makes connection to the database """
    try:
        conn = mysql.connector.connect(user='aws-ml', password='fquGav22QRTaRYhp6n',
                                      database='getabstract',
                                      host='62.12.129.130')
        cursor = conn.cursor()
        
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        print("You are connected!")
    if conn:    
        try:            
            customer_df = pd.read_sql(customer_query, con=conn)

        except Exception as e:
            print("Trace: {}".format(e))

        cursor.close()
        conn.close()
    else:
        print("Connection is not established")
    
    return customer_df


def active_customers():
    """ This function returns only active customers dataframe """
    try:
        active_customers_query= "SELECT \
                            customer.customerId, \
                            customer.languageId, \
                            customer.loginCounter, \
                            subscription.subscriptionTypeId, \
                            product.productType, \
                            product.code, \
                            product.months, \
                            product.isRenew \
                            FROM subscription \
                            INNER JOIN customer \
                            ON subscription.customerId = customer.customerId \
                            INNER JOIN product \
                            ON product.productId = subscription.subscriptionTypeId \
                            WHERE subscription.endDate >= CURDATE()"

        active_customers_dataframe = _db_conn(customer_query=active_customers_query)

        print("Number of records in active_customers_dataframe is {}".format(active_customers_dataframe.shape[0]))
        
    except Exception as e:
        print("trace: {}".format(e))

    return active_customers_dataframe


def pull_customer_data():
#     customer_books_dataframe_ = pd.DataFrame()
    try:
        customers_book_query= 'Select * from {} '.format('customerSummaryIxStatus')
        customers_dataid_dataframe_ = _db_conn(customer_query=customers_book_query)
        print("shape of the cusotmer data is {}".format(customers_dataid_dataframe_.shape))
        
    except Exception as e:
        print("trace: {}".format(e))
        
    return customers_dataid_dataframe_


def drop_books_older_than_5_years(customer_dataframe):
    
    try:
        filtered_year = CURRENT_YEAR - 5
        # Creating year column from the ConsumedAt.
        customer_dataframe['consumedAt_year'] = customer_dataframe['consumedAt'].apply(lambda x: x.year)
        # Dropping rows where year is less than 2010
        customer_dataframe = customer_dataframe.loc[customer_dataframe.consumedAt_year> filtered_year]

        # Dropping year column
        customer_dataframe = customer_dataframe.drop(columns=['consumedAt_year'], axis=1)
        
    except Exception as e:
        print("trace: {}".format(e))

    return customer_dataframe


def get_active_books():
    
    try:
        inactive_summaries_query = "SELECT * \
                                    FROM sumDataBook \
                                    WHERE sumDataBook.active IS TRUE AND \
                                    sumDataBook.dateDeactivated IS NULL"


        # sumDataBook.dataId
        active_books_dataframe = _db_conn(customer_query=inactive_summaries_query)

        # CONSIDERING ONLY FEW COLUMNS
        active_books_dataframe = active_books_dataframe[['dataId', 'sumSourceTypeId', 'dateActivated', 'dateAudioActivated', 
                                                         'dateDeactivated', 'dateAudioDeactivated', 'dateFirstActivation', 
                                                         'dateFirstAudioActivation', 'originalFileName','dateAutomaticAudioActivation']]

        # CREATING DEACTIVATING YEAR FIELD
        active_books_dataframe['dateActivated_year'] = active_books_dataframe['dateActivated'].apply(lambda x: x.year)
        active_books_dataframe['diff_year'] = CURRENT_YEAR - active_books_dataframe.dateActivated_year

        # GIVING WEIGHT TO EACH BOOK, VIDEO, PODCAST, REPORT BASED ON THE YEARS
        active_books_dataframe.loc[((active_books_dataframe.sumSourceTypeId==1) & (active_books_dataframe.diff_year >= 4)), 'book_weight'] = 1
        active_books_dataframe.loc[(((active_books_dataframe.sumSourceTypeId==3)|(active_books_dataframe.sumSourceTypeId==5)) & 
                                    (active_books_dataframe.diff_year >= 1)), 'book_weight'] = 1


        active_books_dataframe.loc[((active_books_dataframe.sumSourceTypeId==1) & ((2 < active_books_dataframe.diff_year) & 
                                                                                   (active_books_dataframe.diff_year<4))), 'book_weight'] = 2

        active_books_dataframe.loc[(((active_books_dataframe.sumSourceTypeId==3)|(active_books_dataframe.sumSourceTypeId==5)) & 
                                    (0 <active_books_dataframe.diff_year)), 'book_weight'] = 2

        active_books_dataframe.loc[((active_books_dataframe.sumSourceTypeId==1) & 
                                    (active_books_dataframe.diff_year <= 2)), 'book_weight'] = 3


        active_books_dataframe.loc[(((active_books_dataframe.sumSourceTypeId==3)|(active_books_dataframe.sumSourceTypeId==5)) &
                                    (active_books_dataframe.diff_year == 0)), 'book_weight'] = 3

        active_books_dataframe.loc[(active_books_dataframe.sumSourceTypeId.isin([2,4,6]), 'book_weight')] = 1

        active_books_dataframe = active_books_dataframe[['dataId', 'book_weight']]

        # Displays the total Number of inactive books
        print("Total number of active books {}.".format(len(active_books_dataframe)))
        
    except Exception as e:
        print("trace: {}".format(e))
    
    return active_books_dataframe


def data_preprocessing(dataframe):
    
    try:
        dataframe.loc[((dataframe['likedAt'].isnull() == False)), 'liked_rating'] = 4
        dataframe['liked_rating'] = dataframe['liked_rating'].fillna(0)

        dataframe.loc[((dataframe['bookmarkedAt'].isnull() == False)), 'bookmarked_rating'] = 2
        dataframe['bookmarked_rating'] = dataframe['bookmarked_rating'].fillna(0)

        dataframe.loc[((dataframe['consumedAt'].isnull() == False)), 'consumedAt_rating'] = 3
        dataframe['consumedAt_rating'] = dataframe['consumedAt_rating'].fillna(0)

        # Multiplying consumedAt_rating with readingProgress to get the final rating
        dataframe['consumedAt_rating'] = dataframe['consumedAt_rating']*dataframe['readingProgress']


        # Creating final rating by adding all the three newly derived variables - liked_rating, bookmarked_rating, consumedAt_rating
        dataframe['rating'] = dataframe['liked_rating'] + \
                                dataframe['bookmarked_rating']  + \
                                dataframe['consumedAt_rating'] + \
                                dataframe['book_weight']
    except Exception as e:
        print("trace: {}".format(e))

    return dataframe


def test_train_split(dataframe):
    
    try:
        count_ratings_per_user = Counter(dataframe['customerId'])
        ## list of users who rated more than 2 books
        more_than_2 = [i[0] for i in count_ratings_per_user.items() if i[1] > 2]

        ## list of users who rated less than 2 books- this dataset can be later appended to train data
        less_than_2 = [i[0] for i in count_ratings_per_user.items() if i[1] <= 2]

        # dataframe containing information about users who rated more than 2 books
        filtering_df_more_than_2 = dataframe.loc[dataframe.customerId.isin(more_than_2)]
        print("Number of rows after dataframe with users rated MORE THAN 2 BOOKS is {}.".format(filtering_df_more_than_2.shape))


        # dataframe containing information about users who rated less than 2 books
        filtering_df_less_than_2 = dataframe.loc[dataframe.customerId.isin(less_than_2)]
        print("Number of rows after dataframe with users rated LESS THAN 2 BOOKS is {}.".format(filtering_df_less_than_2.shape))


        training, testing = train_test_split(filtering_df_more_than_2, shuffle=True, test_size=0.40, train_size=0.60)

    except Exception as e:
        print("trace: {}".format(e))
        
    return training, testing
    
    
    
def pearson_correlation(matrix):
    
    try:
        ## Finding correlation using matrix multiplication (Model)
        corr_matrix = np.corrcoef(matrix.T)
        corrMatrix_train = pd.DataFrame(corr_matrix, index=matrix.columns, columns=matrix.columns)

        corrMatrix_train_ = corrMatrix_train.rename_axis('dataId_y', axis=1)
        corrMatrix_train_.index.names = ['dataId_x']
        corrMatrix_train_ = corrMatrix_train_.stack().reset_index()
        corrMatrix_train_ = corrMatrix_train_.rename(columns={0: 'similarity_score'})

        ## Considering only books which have similarity score more than 0.25
        filtered_correlated_results_gt = corrMatrix_train_.loc[(corrMatrix_train_.similarity_score> 0.25) & 
                                                          (corrMatrix_train_.dataId_x!=corrMatrix_train_.dataId_y)]

    except Exception as e:
        print("trace: {}".format(e))
        
    return filtered_correlated_results_gt


def _merge_train_sim_df(train, item_item_sim_df):
    
    try:
        
        # Merging train data with corremodel_evaluation dataframe to get the recommendation for each user
        dataframe = pd.merge(train[['customerId', 'dataId']], item_item_sim_df, left_on='dataId', right_on='dataId_x')
        # Droping the dataId and dataId_x columns as they are actual columns. So just keeping recommended books list
        dataframe = dataframe.drop(columns=['dataId', 'dataId_x'], axis=1)
        # Renaming the dataId_y column to recommended_books
        dataframe = dataframe.rename(columns={'dataId_y': 'recommended_books'})
    
    except Exception as e:
        print("trace: {}".format(e))
        
    return dataframe 


    
def _select_top_n_books(dataframe, N):
    
    try:
        # Select top 30 books based on similarity score
        dataframe = dataframe.sort_values('similarity_score',ascending = False).groupby('customerId').head(N)
        print("Number of rows in a dataframe recommended_result is {}.".format(dataframe.shape[0]))
        
    except Exception as e:
        print("trace: {}".format(e))
    return dataframe


def _recommend_latest_books(dataframe):
    
    try:
        recommended_results_ = dataframe.groupby(["customerId"]).apply(lambda x: x.sort_values(["recommended_books"], ascending = False)).reset_index(drop=True)

        # Below code only considers latest books with the highest score
        similarity_score_max = recommended_results_.groupby(['customerId', "recommended_books"])['similarity_score'].transform(max)
        recommended_results_ = recommended_results_.loc[recommended_results_['similarity_score'] == similarity_score_max]
    
    except Exception as e:
        print("trace: {}".format(e))
    
    return recommended_results_


def _groupby_books_by_customers(dataframe):
    
    try:
    ## Concatenating books list by customerId
        recommended_books_list = dataframe[['customerId', 'recommended_books']].groupby('customerId').agg(lambda x: x.tolist()).reset_index()
    
    except Exception as e:
        print("trace: {}".format(e))
        
    return recommended_books_list



def _merge_test_recommended_books(test, dataframe):
    
    try:
        
        test = test[['customerId', 'dataId']].groupby('customerId').agg(lambda x: x.tolist())
        test = test.reset_index()
        # merging test and recommended dataframe to calculate accuracy
        merged_df = pd.merge(test, dataframe, on='customerId', how='inner')
        print("Shape of the merged dataframe {}.".format(merged_df.shape))
        
    except Exception as e:
        print("trace: {}".format(e))
    
    return merged_df


def _count_books_predicted_correctly(dataframe):
    
    try:
        dataframe['flag'] = dataframe[['dataId','recommended_books']].apply(lambda x : len(set.intersection(*map(set,list(x)))),axis=1)
  
    except Exception as e:
        print("trace: {}".format(e))
    
    return dataframe


def _accuracy(dataframe):
    accuracy = 0
    try:
        accuracy = len(dataframe.loc[dataframe['flag']>=1])/len(merged_df)
        
    except Exception as e:
        print("trace: {}".format(e))
        
    return round(accuracy,2)*100



def main():
    
    try:
        acive_customer_dataframe = active_customers()
        customer_books_dataframe = pull_customer_data()

        # DROPPING BOOKS OLDER THAN 5 YEARS (NOTE: THIS NEEDS TO BE CHANGED)
        customer_books_dataframe = drop_books_older_than_5_years(customer_dataframe=customer_books_dataframe)

        # PULL ACTIVE BOOKS DATAFRAME
        active_books_dataframe = get_active_books()

        # CONSIDER ONLY ACTIVE CUSTOMERS
        customers_books_dataframe =   customer_books_dataframe.loc[customer_books_dataframe.customerId.isin(acive_customer_dataframe.customerId)]
        print("The number of records in customers book dataframe {}".format(customers_books_dataframe.shape[0]))

        # CONSIDER ONLY ACTIVE BOOKS
        customers_books_dataframe = pd.merge(customers_books_dataframe, active_books_dataframe, on='dataId', how='inner')
        print("The number of records in customers book dataframe {}".format(customers_books_dataframe.shape[0]))


        # CLEANING DATAFRAME
        customers_books_dataframe = data_preprocessing(dataframe=customers_books_dataframe)
        user_book = customers_books_dataframe[["customerId", "dataId", "rating", "consumedAt"]]


        train_, test_ = test_train_split(dataframe=user_book)
        print("Shape of train data is {}.".format(train_.shape))
        print("Shape of test data is {}.".format(test_.shape))


        # PIVOT TABLE
        userbybook_train_ = pd.pivot_table(train_, index = "customerId", columns="dataId", values = "rating", fill_value=0)

        # PEARSON CORRELATION
        filtered_correlated_results_gt = pearson_correlation(userbybook_train_)

        # MERGE TRAIN DATASET WITH CORRELATION MATRIX
        recommended_result = _merge_train_sim_df(train=train_, item_item_sim_df=filtered_correlated_results_gt)

        # SELECT TOP 40 books
        recomemnded_result = _select_top_n_books(dataframe=recommended_result, N=40)

        # RECOMMEND LATEST BOOKS
        recomemnded_result = _recommend_latest_books(dataframe=recomemnded_result)

        # CREATE A LIST OF BOOKS BY CUSTOMERS
        recommended_books_list = _groupby_books_by_customers(dataframe=recomemnded_result)

        # ADD THE BOOKS READ BY THE CUSTOEMRS(ACTUAL DATAID)
        merged_df = _merge_test_recommended_books(test=test_, dataframe=recommended_books_list)

        # CHECKING IF RECOMMENDATION OVERLAPS WITH THE DATAIDS READ BY THE CUSTOMERS
        merged_df =  _count_books_predicted_correctly(dataframe=merged_df)

        # CALCUALTE ACCURACY
        accuracy_pearson = _accuracy(dataframe=merged_df)

        print("*************************************************")
        print("Accuracy of the Pearson Correlation model is {} %".format(accuracy_pearson))
        print("*************************************************")

        recommended_result = recomemnded_result[['customerId', 'recommended_books', 'similarity_score']]
    
    except Exception as e:
        print("trace: {}".format(e))

    return recomemnded_result


if __name__=="__main__": 
    recommender_output = main()