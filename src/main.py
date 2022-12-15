import easyfl

if __name__ == "__main__":
    config = easyfl.load_config("./config/config.yaml")
    
    easyfl.init(config)

    easyfl.run()
