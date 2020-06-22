import pickle

import pymongo

if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://localhost:27017/')

    db_list = myclient.list_database_names()
    db_name = 'Jobmd'
    corpora_label = []

    level = {
        '应届生': 1,
        '1-3年': 2,
        '3-5年': 3,
        '5-10年': 4,
        '10年以上': 5
    }

    working_exp_dict = {'经验不限': 0,
                        '在读学生': 0,
                        '应届生': 0,  # level 1
                        '1-3年': 0,  # level 2
                        '3-5年': 0,  # level 3
                        '5-10年': 0,  # level 4
                        '10年以上': 0  # level 5
                        }

    if db_name in db_list:
        jobmd_db = myclient[db_name]
        collection_names = jobmd_db.list_collection_names()
        for collection_name in collection_names:
            if '医院临床医疗' in collection_name:
                my_collection = jobmd_db.get_collection(collection_name)
                for job_item in my_collection.find():
                    job_desc = job_item['jobDesc']
                    working_exp = job_item['workingExp']
                    if len(job_desc) > 15 and working_exp in level.keys():
                        working_exp_dict[working_exp] += 1
                        corpora_label.append((job_desc, level[working_exp]))

    print(working_exp_dict)
    with open('./corpora_data/corpora_label.bin', "wb") as f:
        pickle.dump(corpora_label, f)
