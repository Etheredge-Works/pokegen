from ts.torch_handler.base_handler import BaseHandler
class Handler(BaseHandler):
    def __init__(self, *args, **kwargs):
        super(Handler, self).__init__(*args, **kwargs)
        self.model = self.model.cuda()
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, data):
        data = data.to(self.device)
        output = self.model(data)
        return output.cpu().detach().numpy()

    def predict_batch(self, data):
        data = data.to(self.device)
        output = self.model(data)
        return output.cpu().detach().numpy()