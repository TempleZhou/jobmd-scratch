import pymongo

if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://localhost:27017/')

    db_list = myclient.list_database_names()
    db_name = 'Jobmd'
    corpora = []

    if db_name in db_list:
        jobmd_db = myclient[db_name]
        collection_names = jobmd_db.list_collection_names()
        for collection_name in collection_names:
            if '医院临床医疗' in collection_name:
                my_collection = jobmd_db.get_collection(collection_name)
                for job_item in my_collection.find():
                    job_desc = job_item['jobDesc']
                    if len(job_desc) > 15:
                        corpora.append(job_desc+"\n")

    with open('corpora_data/corpora.txt', 'w') as f:
        f.writelines(corpora)
