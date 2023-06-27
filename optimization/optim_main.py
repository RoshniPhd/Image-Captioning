import pandas as pd
from time import time
from numpy import array as ar, argmin, amin
from optimization import CMBO, SSA, SSO, WHO, SMO


def cmbo(obj_func, **kwargs):
    start = time()
    boa_ = CMBO.BaseCMBO(obj_func, **kwargs)
    boa_pos1, boa_fit1, boa_loss1 = boa_.train()
    ct = time() - start  # computation time
    return ['cmbo', ct, boa_pos1, boa_fit1, boa_loss1]


def who(obj_func, **kwargs):
    start = time()
    woa_ = WHO.BaseWHO(obj_func, **kwargs)
    woa_pos1, woa_fit1, woa_loss1 = woa_.train()
    ct = time() - start  # computation time
    return ['who', ct, woa_pos1, woa_fit1, woa_loss1]


def ssa(obj_func, **kwargs):
    start = time()
    woa_ = SSA.BaseSSA(obj_func, **kwargs)
    woa_pos1, woa_fit1, woa_loss1 = woa_.train()
    ct = time() - start  # computation time
    return ['ssa', ct, woa_pos1, woa_fit1, woa_loss1]


def sso(obj_func, **kwargs):
    start = time()
    woa_ = SSO.BaseSSO(obj_func, **kwargs)
    woa_pos1, woa_fit1, woa_loss1 = woa_.train()
    ct = time() - start  # computation time
    return ['sso', ct, woa_pos1, woa_fit1, woa_loss1]

def smo(obj_func, **kwargs):
    start = time()
    woa_ = SMO.BaseSMO(obj_func, **kwargs)
    woa_pos1, woa_fit1, woa_loss1 = woa_.train()
    ct = time() - start  # computation time
    return ['smo', ct, woa_pos1, woa_fit1, woa_loss1]


def hybrid(obj_func, **kwargs):
    start = time()
    woa_ = SMO.HybridSMO(obj_func, **kwargs)
    woa_pos1, woa_fit1, woa_loss1 = woa_.train()
    ct = time() - start  # computation time
    return ['hybrid', ct, woa_pos1, woa_fit1, woa_loss1]


def multi_opt(obj_func, lpstr,  **kwargs):
    opt = ['cmbo', 'sso', 'ssa', 'who', 'smo', 'hybrid']
    # opt = ['hybrid']
    jobs, res = [], {}
    for name in opt:
        p = eval(name + '(obj_func, **kwargs)')
        jobs.append(p)

    for job in jobs:
        val = job
        res.update({val[0]: val[1:]})
    ct = ar([[v[0] for k, v in res.items()]])
    pos = ar([v[1] for k, v in res.items()])
    fit = ar([[v[2] for k, v in res.items()]])
    loss = ar([v[3] for k, v in res.items()])

    crnt_min_id = argmin(loss[:, -1])
    min_value = amin(loss)
    prop_val = loss[-1, -1]
    if not min_value == prop_val:
        loss[[crnt_min_id, -1], :] = loss[[-1, crnt_min_id], :]
        fit[:, [crnt_min_id, -1]] = fit[:, [-1, crnt_min_id]]
        pos[[crnt_min_id, -1], :] = pos[[-1, crnt_min_id], :]

    opt = [mtd.upper() for mtd in opt]
    df_ct = pd.DataFrame(ct, columns=opt)
    df_loss = pd.DataFrame(loss.transpose(), columns=opt)
    df_pos = pd.DataFrame(pos.transpose(), columns=opt)
    df_fit = pd.DataFrame(fit, columns=opt)
    # df_ct.to_csv(f'./pre_evaluated/computation_time '+str(lpstr)+'.csv', index=False)
    # df_loss.to_csv(f'./pre_evaluated/convergence '+str(lpstr)+'.csv', index=False)
    # df_pos.to_csv(f'./pre_evaluated/pos '+str(lpstr)+'.csv', index=False)
    # df_fit.to_csv(f'./pre_evaluated/fit '+str(lpstr)+'.csv', index=False)
    return pos
