def get_config(name, args=None):
    if '+' in name:
        pass
    elif 'cvprlc' in name:
        from .opt_CVPR_LandCover import Config
    elif 'dlrsd' in name:
        from .opt_DLRSD import Config
    elif 'isprspd' in name:
        from .opt_ISPRS_Postdam import Config

    if args is not None:
        mOptions = Config(args)
    else:
        mOptions = Config()

    return mOptions

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR+'/Tools/dltoos')
