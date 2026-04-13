import logging
import os
from typing import BinaryIO, List, Any, Union
import io
from minio import Minio
from minio.error import S3Error
import json
import ast



if __name__=='__main__':
    mc=Minio(
                endpoint='localhost:9000',
                access_key='minio',
                secret_key='minio123',
                secure=False
            )
    bucket_name='pdf-processing'
    print(mc.bucket_exists(bucket_name))
    
    dict_pdfs={}
    for obj in mc.list_objects(bucket_name, recursive=True, prefix='pdfs'):
        hasher=obj.object_name.split('/')[-2].split('_')[0]
        filename=obj.object_name.split('/')[-1]
        if(hasher in dict_pdfs.values()):
            print(f"repeate: {filename} <-> {hasher}")
        else:
            dict_pdfs[filename]=hasher
            
        
    print(len(dict_pdfs))
    with open('file_hash_comparison.json', 'w') as file:
        json.dump(dict_pdfs, file)
    
    