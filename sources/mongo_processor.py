from pymongo import MongoClient
from bson.objectid import ObjectId
import datetime

class Mongo(object):
  """
  loki db
  """
  def __init__(self, host, port, id, pwd, db_name, collection,mongo_uri):
    self.db_name = db_name
    self.collection = collection
    self.client = MongoClient(host, int(port))
    self.client.the_database.authenticate(id,
                                          pwd,
                                          source=db_name)


  def find_item(self, condition=None, data=None):
    result = self.client[self.db_name][self.collection].find({})
    return result

  def find_user_item(self, condition=None, data=None):
    result = self.client[self.db_name][self.collection].find({"_id": ObjectId(data)})
    result = self.client[self.db_name][self.collection].find(data)
    return result


  def find_phase(self, condition=None, datas=None, id=None):
    result = self.client[self.db_name][self.collection].find({"phase": 3,"user_id":{'$nin': [data for data in datas]},
                                                              })
    return result

  def find_data_id(self, user_ids=None, datas=None):
    result = self.client[self.db_name][self.collection].find({'phase':3, 'user_id':{'$nin': [data for data in user_ids]},
                                                              'data_id':datas})
    return result

  def find_user_to_ids(self, condition=None, data=None):
    result = self.client[self.db_name][self.collection].find({"user_id": ObjectId(data)}, {'phase':3})
    return result

  def find_item2(self, data=None):
    result = self.client[self.db_name][self.collection].find(data)
    return result

  def find_item3(self, data=None):
    result = self.client[self.db_name][self.collection].find(data)
    return result

  def find_number(self, condition=None, db_name=None, collection=None, data=None):
    result = self.client[db_name][collection].find({"_number": number})
    return result

  def find_user_id(self, data=None):
    result = self.client[self.db_name][self.collection].find(data)
    return result

  def find_id(self, condition=None):
    result = self.client[self.db_name][self.collection].find({}, {"_id": 1})
    return result

  def find_id2(self, idx):
    result = self.client[self.db_name][self.collection].find({"idx": idx})[0]
    return result

  def find_id3(self, idx):
    result = self.client[self.db_name][self.collection].find({"idx": idx})[0]
    return result
    
  def find_data(self, condition=None, db_name=None, collection=None, date=None):
    result = self.client[db_name][collection].find({'phase':3, 'createdAt': {'$gte': date}})
    return result

  def insert_item_one(self, data, db_name=None, collection_name=None):
    result = self.client[db_name][collection_name].insert_one(data).inserted_id
    return result

  def update_item_one(self, condition=None, update_value=None, db_name=None, collection_name=None):
    result = self.client[db_name][collection_name].update(filter=condition, update=update_value, upsert=True)
    return result

  def count(self, db_name=None, collection=None):
    result = self.client[db_name][collection].count()
    return result

  def update_ex(self, condition=None, update_value=None, db_name=None, collection_name=None):
    result = self.client[self.db_name][self.collection].update(condition, {'$set': update_value}, upsert=True)
    return result
