from tensorflow.keras import layers, Model, callbacks

class AnomalyAutoencoder:
    def __init__(self, input_dim, encoding_dim, cfg):
        inp = layers.Input(shape=(input_dim,))
        x = layers.Dense(cfg['hidden_dim'], activation='relu')(inp)
        encoded = layers.Dense(encoding_dim, activation='relu')(x)
        x = layers.Dense(cfg['hidden_dim'], activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(x)
        self.model = Model(inp, decoded)
        self.model.compile(optimizer=cfg['optimizer'], loss=cfg['loss'])
        self.callbacks = [
            callbacks.EarlyStopping(monitor='val_loss', patience=cfg['patience'], restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=cfg['lr_factor'], patience=cfg['lr_patience'])
        ]

    def train(self, X, val_split, epochs=100, batch=32):
        return self.model.fit(X, X, epochs=epochs, batch_size=batch,
                              validation_split=val_split, callbacks=self.callbacks)

    def detect(self, X, thresh):
        recon = self.model.predict(X)
        mse = ((X - recon)**2).mean(axis=1)
        return mse, mse > thresh

    def save(self, path: str):
        # сохраняем архитектуру и веса
        self.model.save(path)