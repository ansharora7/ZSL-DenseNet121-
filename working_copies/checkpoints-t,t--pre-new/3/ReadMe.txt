lr=3e-3
epochs=40
optimizer= self.optimizer  = optim.SGD(list(self.model.parameters()) + list(self.vae.parameters()), lr=3e-3, momentum=0.9, weight_decay=0.0001)
           self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')
total_loss does not contain vae loss