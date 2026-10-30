'''
Stand-in planner workers for test_planner.py. Each runs in a spawned process in place of
planner_process._planner_worker_main, with the same arguments and message protocol, and
answers a request with the request's goal as a list. Only the standard library is imported
here, so a stand-in starts in a fraction of a second.
'''
import os
import time


def _answer(request):
    return {'id': request['id'], 'type': 'result', 'result': [float(x) for x in request['goal']]}


def _serve(request_queue, response_queue, respond):
    while True:
        request = request_queue.get()
        if request.get('command') == 'shutdown':
            return
        respond(request, response_queue)


def echo(init_args, request_queue, response_queue):
    '''Ready at once; answers every request'''
    response_queue.put({'type': 'ready'})
    _serve(request_queue, response_queue,
           lambda request, responses: responses.put(_answer(request)))


def ready_after_a_second(init_args, request_queue, response_queue):
    '''Ready after 1 s, as a slow model build; then answers every request'''
    time.sleep(1.0)
    echo(init_args, request_queue, response_queue)


def stale_answer_first(init_args, request_queue, response_queue):
    '''Before each answer, sends one for the request before it, as a late reply would'''
    def respond(request, responses):
        responses.put({'id': request['id'] - 1, 'type': 'result', 'result': 'stale'})
        responses.put(_answer(request))

    response_queue.put({'type': 'ready'})
    _serve(request_queue, response_queue, respond)


def dies_at_start(init_args, request_queue, response_queue):
    '''Exits with code 3 before it is ready'''
    os._exit(3)


def never_answers(init_args, request_queue, response_queue):
    '''Ready at once; takes requests without answering them, until told to shut down'''
    response_queue.put({'type': 'ready'})
    _serve(request_queue, response_queue, lambda request, responses: None)


def echo_after_a_while(init_args, request_queue, response_queue):
    """Ready at once; answers each request 0.2 s after taking it"""
    def respond(request, responses):
        time.sleep(0.2)
        responses.put(_answer(request))

    response_queue.put({'type': 'ready'})
    _serve(request_queue, response_queue, respond)
