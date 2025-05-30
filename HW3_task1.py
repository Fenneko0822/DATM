import numpy as np


def z_normalize(ts):
    mean = np.mean(ts)
    std = np.std(ts)
    return (ts - mean) / std if std > 0 else np.zeros_like(ts)


def compute_distance_profile(ts, query):
    m = len(query)
    dp_len = len(ts) - m + 1
    dp = np.empty(dp_len)
    query = z_normalize(query)

    for i in range(dp_len):
        dp[i] = np.linalg.norm(z_normalize(ts[i : i + m]) - query)

    return dp


def apply_exclusion(dp, index, m, exclude):
    # TODO: apply the exclusion zone around the indices in dp
    exculsion_zone = int((m * exclude) // 2)
    start = max(0, index - exculsion_zone)
    end = min(len(dp), index + exculsion_zone + 1)
    dp[start:end] = np.inf
        

def matrix_profile(ts, m, exclude=0.0):

    n = len(ts)
    mp = np.full(n - m + 1, np.inf)
    mp_idx = np.full(n - m + 1, -1)

    for i in range(n - m + 1):
        query = ts[i : i + m]
        dp = compute_distance_profile(ts, query)
        apply_exclusion(dp, i, m, exclude)

        min_idx = np.argmin(dp)
        mp_idx[i] = min_idx
        mp[i] = dp[min_idx]

    return mp, mp_idx


def find_discords(ts, window, k):
    """Detects the top-k discords in a time series."""
    mp, mp_idx = matrix_profile(ts, window)
    discords = []

    mp_copy = mp.copy()

    for _ in range(k):
        # TODO 1) Find the index of the highest Matrix Profile value

        if np.all(np.isinf(mp_copy)):
            break

        idx = np.argmax(mp_copy)

        if np.isinf(mp_copy[idx]):
            break

        discords.append(idx)

        # TODO 2) Apply exclusion zone to prevent redundant discords
        # Remember that you can exclude points with -np.inf, since
        # we are searching for the largest values.
        start = max(0, idx)
        end = min(len(mp_copy), idx + window)
        mp_copy[start:end] = -np.inf

    return discords

