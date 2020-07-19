from abc import ABCMeta, abstractmethod

class Step(metaclass=ABCMeta):

    def step(self, x_batch, y_batch):
        '''
        Performs a train/test step for the model
        with input 'x_batch' and labels 'y_batch'.
        '''
        raise NotImplementedError


class LearnRawStep(Step):

    def step(self, x_batch, y_batch):
        logits = threat_model.forward(x_batch)
        loss = threat_model.loss(logits, y_batch)

        loss.backward()

        store(logits, loss, y_batch)

        optimManager.step()


class TestRawStep(Step):

    def step(self, x_batch, y_batch):
        logits = threat_model.forward(x_batch)
        loss = threat_model.loss(logits, y_batch)

        loss.backward()

        store(logits, loss, y_batch)

class LearnAdvStep(Step):

    def step(self, x_batch, y_batch):
        ''' Computes performance based on adv data only.'''
        x_adv = attack.run(x_batch, y_batch)
        logits = threat_model.forward(x_adv)
        loss = threat_model.loss(logits, y_batch)

        loss.backward()

        store(logits, loss, y_batch)

        optimManager.step()


class TestAdvStep(Step):

    def step(self, x_batch, y_batch):
        ''' Computes performance based on adv data only.'''
        x_adv = attack.run(x_batch, y_batch)
        logits = threat_model.forward(x_adv)
        loss = threat_model.loss(logits, y_batch)

        store(logits, loss, y_batch)


class LearnMixedStep(Step):

    def step(self, x_batch, y_batch):
        ''' Computes peformance based on raw and adv data.'''
        logits = threat_model.forward(x_batch)
        loss = threat_model.loss(logits, y_batch)
        loss.backward()

        store(logits, loss, y_batch)
        optimManager.step()

        x_adv = attack.run(x_batch, y_batch)
        logits = threat_model.forward(x_adv)
        loss = threat_model.loss(logits, y_batch)
        loss.backward()

        store(logits, loss, y_batch)
        optimManager.step()


class TestMixedStep(Step):

    def step(self, x_batch, y_batch):
        ''' Computes peformance based on raw and adv data.'''
        logits = threat_model.forward(x_batch)
        loss = threat_model.loss(logits, y_batch)

        store(logits, loss, y_batch)

        x_adv = attack.run(x_batch, y_batch)
        logits = threat_model.forward(x_adv)
        loss = threat_model.loss(logits, y_batch)

        store(logits, loss, y_batch)

class LearnAdvHGDStep(Step):
    ''' There is a difference on how the loss is computed during training and testing procedure.'''
    # TODO: Can I reduce the key difference here and abstract the method?
    def step(self, x_batch, y_batch):
        # 1. Generate adversarial example.
        x_adv = attack.run(x_batch, y_batch)

        # KEY DIFFERENCE (logits) Assume training procedure for now...
        lg_smooth, lg_raw = threat_model.trainForward(x_batch, x_adv)
        loss = threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        store(lg_smooth, loss, y_batch)
        optimManager.step()


class TestAdvHGDStep(Step):

    def step(self, x_batch, y_batch):
        x_adv = attack.run(x_batch, y_batch)

        logits = threat_model.testForward(x_adv)
        loss = threat_model.testLoss(logits, y_batch)

        store(logits, loss, y_batch)
        optimManager.zeroGrad()


class LearnMixedHGDStep(Step):

    def step(self, x_batch, y_batch):
        lg_smooth, lg_raw = threat_model.trainForward(x_batch, x_batch)
        loss = threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        store(lg_smooth, loss, y_batch)
        optimManager.step()

        x_adv = attack.run(x_batch, y_batch)
        lg_smooth, lg_raw = threat_model.trainForward(x_batch, x_adv)
        loss = threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        store(lg_smooth, loss, y_batch)
        optimManager.step()


class TestMixedHGDStep(Step):

    def step(self, x_batch, y_batch):
        logits = threat_model.testForward(x_batch)
        loss = threat_model.testLoss(logits, y_batch)

        store(logits, loss, y_batch)
        optimManager.zeroGrad()

        x_adv = attack.run(x_batch, y_batch)
        logits = threat_model.testForward(x_adv)
        loss = threat_model.testLoss(logits, y_batch)

        store(logits, loss, y_batch)
        optimManager.zeroGrad()

# class RawStep(Step):
#
#     def trainStep(self, x_batch, y_batch):
#         ''' Computes performance based raw data only.'''
#         logits = self.threat_model.forward(x_batch)
#         loss = self.threat_model.loss(logits, y_batch)
#
#         loss.backward()
#
#         self.store(logits, loss, y_batch)
#
#         self.optimManager.step()
#
#     def testStep(self, x_batch, y_batch):
#         ''' Computes performance based raw data only.'''
#         logits = self.threat_model.forward(x_batch)
#         loss = self.threat_model.loss(logits, y_batch)
#
#         self.store(logits, loss, y_batch)

# class AdvStep(DefenseStep):
#
#     def trainStep(self, x_batch, y_batch):
#         ''' Computes performance based on adv data only.'''
#         x_adv = self.attack.run(x_batch, y_batch)
#         logits = self.threat_model.forward(x_adv)
#         loss = self.threat_model.loss(logits, y_batch)
#
#         loss.backward()
#
#         self.store(logits, loss, y_batch)
#
#         self.optimManager.step()
#
#     def testStep(self, x_batch, y_batch):
#         ''' Computes performance based on adv data only.'''
#         x_adv = self.attack.run(x_batch, y_batch)
#         logits = self.threat_model.forward(x_adv)
#         loss = self.threat_model.loss(logits, y_batch)
#
#         self.store(logits, loss, y_batch)

# class MixedStep(DefenseStep):
#
#     def trainStep(self, x_batch, y_batch):
#         ''' Computes peformance based on raw and adv data.'''
#         logits = self.threat_model.forward(x_batch)
#         loss = self.threat_model.loss(logits, y_batch)
#         loss.backward()
#
#         self.store(logits, loss, y_batch)
#         self.optimManager.step()
#
#         x_adv = self.attack.run(x_batch, y_batch)
#         logits = self.threat_model.forward(x_adv)
#         loss = self.threat_model.loss(logits, y_batch)
#         loss.backward()
#
#         self.store(logits, loss, y_batch)
#         self.optimManager.step()
#
#     def testStep(self, x_batch, y_batch):
#         ''' Computes peformance based on raw and adv data.'''
#         logits = self.threat_model.forward(x_batch)
#         loss = self.threat_model.loss(logits, y_batch)
#
#         self.store(logits, loss, y_batch)
#
#         x_adv = self.attack.run(x_batch, y_batch)
#         logits = self.threat_model.forward(x_adv)
#         loss = self.threat_model.loss(logits, y_batch)
#
#         self.store(logits, loss, y_batch)

class AdvHGDStep(HGDStep):
    ''' There is a difference on how the loss is computed during training and testing procedure.'''
    # TODO: Can I reduce the key difference here and abstract the method?
    def trainStep(self, x_batch, y_batch):
        # 1. Generate adversarial example.
        x_adv = self.attack.run(x_batch, y_batch)

        # KEY DIFFERENCE (logits) Assume training procedure for now...
        lg_smooth, lg_raw = self.threat_model.trainForward(x_batch, x_adv)
        loss = self.threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        self.store(lg_smooth, loss, y_batch)
        self.optimManager.step()

    def testStep(self, x_batch, y_batch):
        x_adv = self.attack.run(x_batch, y_batch)

        logits = self.threat_model.testForward(x_adv)
        loss = self.threat_model.testLoss(logits, y_batch)

        self.store(logits, loss, y_batch)
        self.optimManager.zeroGrad()


class MixedHGDStep(HGDStep):

    def trainStep(self, x_batch, y_batch):
        lg_smooth, lg_raw = self.threat_model.trainForward(x_batch, x_batch)
        loss = self.threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        self.store(lg_smooth, loss, y_batch)
        self.optimManager.step()

        x_adv = self.attack.run(x_batch, y_batch)
        lg_smooth, lg_raw = self.threat_model.trainForward(x_batch, x_adv)
        loss = self.threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        self.store(lg_smooth, loss, y_batch)
        self.optimManager.step()

    def testStep(self, x_batch, y_batch):
        logits = self.threat_model.testForward(x_batch)
        loss = self.threat_model.testLoss(logits, y_batch)

        self.store(logits, loss, y_batch)
        self.optimManager.zeroGrad()

        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.testForward(x_adv)
        loss = self.threat_model.testLoss(logits, y_batch)

        self.store(logits, loss, y_batch)
        self.optimManager.zeroGrad()
