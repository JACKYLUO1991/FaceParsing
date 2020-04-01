from utils import *
import time
from metrics import SegMetric


class Verifier(object):

    def __init__(self, data_loader, config):

        self.data_loader = data_loader
        self.classes = config.classes
        self.arch = config.arch

    def validation(self, G):
        G.eval()
        time_meter = AverageMeter()

        # Model loading
        metrics = SegMetric(n_classes=self.classes)
        metrics.reset()

        for index, (images, labels) in enumerate(self.data_loader):
            if (index + 1) % 100 == 0:
                print('%d processd' % (index + 1))

            images = images.cuda()
            labels = labels.cuda()
            h, w = labels.size()[1], labels.size()[2]

            torch.cuda.synchronize()
            tic = time.perf_counter()

            with torch.no_grad():
                outputs = G(images)
                # Whether or not multi branch?
                if self.arch == 'CE2P' or 'FaceParseNet' in self.arch:
                    outputs = outputs[0][-1]

                outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
                pred = outputs.data.max(1)[1].cpu().numpy()  # Matrix index
                gt = labels.cpu().numpy()
                metrics.update(gt, pred)

            torch.cuda.synchronize()
            time_meter.update(time.perf_counter() - tic)

        print("Inference Time: {:.4f}s".format(
            time_meter.average() / images.size(0)))

        # mIoU metric
        score = metrics.get_scores()[0]

        return score["Mean IoU : \t"]
