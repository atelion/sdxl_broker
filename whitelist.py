import bittensor as bt

def get_big_stake_uids(stake_list: list, min_stake: float):
    try:
        big_stake_uids = [k for k in range(len(stake_list)) if stake_list[k] >= min_stake]
        return big_stake_uids
    except:
        return []
    
def get_whitelist(netuid: int, min_burn: float):
    whitelist = []
    metagraph = bt.subtensor('finney').metagraph(netuid=netuid)
    axons = metagraph.axons
    hotkeys = metagraph.hotkeys
    stakes = metagraph.S
    big_stake_uids = get_big_stake_uids(stakes, min_burn)
    big_stake_hotkeys = [hotkeys[uid] for uid in big_stake_uids]
    big_stake_axons = [axons[uid] for uid in big_stake_uids]
    big_stake_ips = [axon.ip for axon in big_stake_axons]
    return big_stake_hotkeys, big_stake_ips

def check_white_dendrite(netuid: int, min_burn: float, dendrite: bt.chain_data.AxonInfo):
    big_stake_hotkeys, big_stake_ips = get_whitelist(netuid, min_burn)
    if dendrite.hotkey in big_stake_hotkeys and dendrite.ip in big_stake_ips:
        return True
    return False
if __name__ == "__main__":
    dendrite = bt.chain_data.AxonInfo(ip="194.163.145.217", version=1, port=0, ip_type=4, coldkey="",  hotkey="5DvTpiniW9s3APmHRYn8FroUWyfnLtrsid5Mtn5EwMXHN2ed")
    print(check_white_dendrite(28, 4096, dendrite))