import time


def Eval_Model(env, rr_matcher, vr_matcher, selected_day):
    t0 = time.time()
    done = False
    env.repeat(selected_day)
    print(f'the selected day is {env.get_time()}')

    Assigned_Request, Loss_Request, Distance_Driven, Waiting_Time, Revenue = 0, 0, 0, 0, 0
    while not done:
        dispatch_action = {}
        rr_graph = env.get_rr_match_graph()
        if rr_graph is not None and len(rr_graph.edges) > 0:
            _, rr_decision = rr_matcher.get_rr_match_decision(rr_graph)
            env.do_rr_match(rr_decision)

            vr_graph = env.get_vr_match_graph()
            if vr_graph is not None and len(vr_graph.edges) > 0:
                _, vr_decision = vr_matcher.get_vr_match_decision(vr_graph)
                dispatch_action = env.do_vr_match(vr_decision)

        done, assigned_request, loss_request, distance_driven, waiting_time, revenue = env.step(dispatch_action)
        Assigned_Request += assigned_request
        Loss_Request += loss_request
        Distance_Driven += distance_driven
        Waiting_Time += waiting_time
        Revenue += revenue

    tf = time.time()
    # print(f'Evaluation time: {tf - t0}')
    # print(f'Total assigned requests: {Assigned_Request}')
    # print(f'Total loss requests: {Loss_Request}')
    # print(f'Total distance driven: {Distance_Driven}')
    # print(f'Total waiting time: {Waiting_Time}')
    # print(f'Total revenue: {Revenue}')

    return Assigned_Request, Distance_Driven, Waiting_Time, Revenue
