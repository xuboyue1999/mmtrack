from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()


    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.rgbt234_path = ''
    settings.lasher_path = ''
    settings.gtot_path = ''
    settings.depthtrack_path = ''
    settings.rgbd1k_path = ''
    settings.visevent_path = ''

    return settings
