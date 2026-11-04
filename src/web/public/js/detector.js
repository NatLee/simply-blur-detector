// The CNN, run by ONNX Runtime (Web in the browser, the same package under Node.js in the tests).
import { toModelInput } from './imageops.js';

export const MODEL_CARD_FORMAT = 1;

export class BlurDetector {
  /**
   * @param ort    the onnxruntime-web module
   * @param model  URL or bytes of the ONNX file
   * @param card   the model card written by `blur-detector export`
   */
  static async load(ort, model, card, sessionOptions = {}) {
    if (card.format !== MODEL_CARD_FORMAT) {
      throw new Error(`model card format ${card.format} is not supported (expected ${MODEL_CARD_FORMAT})`);
    }
    const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], ...sessionOptions });
    return new BlurDetector(ort, session, card);
  }

  constructor(ort, session, card) {
    this.ort = ort;
    this.session = session;
    this.size = card.input.size;
    this.blurIndex = card.classes.indexOf('Blur');
    this.classCount = card.classes.length;
  }

  /** P(blur) of each image ({width, height, data} RGBA), prepared exactly like the training data. */
  async predict(images) {
    const pixels = this.size * this.size * 3;
    const batch = new Float32Array(images.length * pixels);
    images.forEach((img, i) => batch.set(toModelInput(img, this.size), i * pixels));
    const input = new this.ort.Tensor('float32', batch, [images.length, this.size, this.size, 3]);
    const outputs = await this.session.run({ [this.session.inputNames[0]]: input });
    const probabilities = outputs[this.session.outputNames[0]].data;
    return images.map((_, i) => probabilities[i * this.classCount + this.blurIndex]);
  }
}
