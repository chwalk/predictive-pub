import logging 
import numpy as np 
import azure.functions as func 
 

bell_curve = None 
 

def create():
    global bell_curve 
    bell_curve = np.random.normal(0, 20, 10000) 
    return func.HttpResponse(f"Created the bell curve.") 


def show_point():
    size = bell_curve.size 
    index = np.random.randint(0, size) 
    height = bell_curve[index] 
    logging.info(f"The size of the bell curve is {size}. Found point ({index}, {height}).") 
    return func.HttpResponse(f"Found point ({index}, {height}).") 


def handle_request(request: func.HttpRequest) -> func.HttpResponse: 
    command = request.params.get('command') 
    if not command: 
        try:
            req_body = request.get_json() 
        except ValueError: 
            pass
        else: 
            command = req_body.get('command') 

    if command: 
        if command == 'create':
            return create()
        elif command == 'show_point':
            return show_point()
        else:
            return func.HttpResponse(f"Command {command} not found.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a command in the query string or in the request body.",
             status_code=200
        )


def main(req: func.HttpRequest) -> func.HttpResponse: 
    logging.info('Received request.') 
    return handle_request(req) 
