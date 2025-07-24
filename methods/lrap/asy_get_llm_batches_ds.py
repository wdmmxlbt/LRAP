import asyncio
from openai import AsyncOpenAI
import time
import aiohttp


BATCH_SIZE = 30  

API_KEY = "your_api_key_here"  # Replace with your actual API key

error_count = 0
error_lock = asyncio.Lock()

base_url = "your_api_base_url_here"  # Replace with your actual API base URL



async def async_process_batch(dataset,session, futures, batch, api_key, idx, query, max_retries=3):
    retries = 0
    global error_count  
    mapping = {
        "clinc": "You are an expert in the field of task-oriented dialogue systems, with a clear and precise understanding of user intent classification. Please help with classifying user intents in this domain.",
        "banking": "You are an expert in the banking domain with a clear and precise understanding of user intent classification in this field. Please help with user intent classification.",
        "stackoverflow": "You possess a deep and precise expertise in the domain of natural language and programming language integration, particularly in classifying user intents. Please help with user intent classification."
    }
    while retries < max_retries:
        async with session.post(
            f"{base_url}/chat/completions", 
            json={
                "model": "deepseek-v3-250324", 
                # "model":"deepseek-r1-250120",
                "messages": [
                    {"role": "system", "content": mapping[dataset]},
                    {"role": "user", "content": query}
                ]
            },
            headers={"Authorization": f"Bearer {api_key}"}  
        ) as response:
            if response.status == 200:
                completion = await response.json()
                new_response = completion["choices"][0]["message"]["content"]
                futures[idx] = new_response
             
                return  
            else:
                retries += 1
               
                await asyncio.sleep(2) 

    futures[idx] = "None"

    async with error_lock:
        error_count += 1
    print(f"Request failed after {max_retries} retries. Setting futures[{idx}] to 'None'.")

async def async_process_queries(dataset,queries):
    all_results = {}
    num_batches = (len(queries) + BATCH_SIZE - 1) // BATCH_SIZE  
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_batches):
            batch = {k: queries[k] for k in list(queries.keys())[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]}  
            futures = {} 
            api_key = API_KEY
            tasks = [async_process_batch(dataset,session,futures,batch, api_key, idx, query) for idx,query in batch.items()]
            await asyncio.gather(*tasks)
            while len(futures) < len(batch):
                await asyncio.sleep(1)
            all_results.update(futures)
    print(all_results)
    return all_results



async def get_query(dataset,train_semi_dataset,queries):
    
    start_time = time.time() 
    results = await async_process_queries(dataset,queries)
    end_time = time.time()  
    for idx,result in results.items():
        print(idx)
        print(f'sentence:{train_semi_dataset[idx]["input_text"]},class:{train_semi_dataset[idx]["label_id_true"]}')
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds, accummulated error: {error_count}")
    return results

async def get_1_query(dataset,queries):
    
    start_time = time.time() 
    results = await async_process_queries(dataset,queries)
    end_time = time.time() 
    for idx,result in results.items():
        print(idx)
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds, accummulated error: {error_count}")
    return results

query = {
    "idx1": "What is the capital of France?"
}

asyncio.run(get_1_query("stackoverflow",query))
print("len dict",len(query))
print("count",error_count)