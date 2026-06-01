import * as tf from '@tensorflow/tfjs';

// ============================================================
// ARQUITETURA: Transfer Learning com MobileNet (4 Classes)
// ============================================================

const IMAGE_SIZE = 224; // Tamanho que o MobileNet espera
let mobileNet = null;   // Rede pré-treinada (extrator de features)
let classifier = null;  // Nossa camada de classificação (treinável)

// Estruturas de dados organizadas para as 4 classes
let classEmbeddings = [[], [], [], []];
let classImages = [[], [], [], []];
let testImageTensor = null;

const classNames = ['Mario', 'Sonic', 'Kratos', 'Outros/Fundo'];

// ---- Mapeamento dos Elementos da UI ----
const uiElements = [1, 2, 3, 4].map(num => ({
  upload: document.getElementById(`uploadClass${num}`),
  btn: document.getElementById(`btnClass${num}`),
  preview: document.getElementById(`previewClass${num}`),
  count: document.getElementById(`countClass${num}`)
}));

const trainBtn = document.getElementById('trainBtn');
const trainingStatus = document.getElementById('trainingStatus');
const trainProgress = document.getElementById('trainProgress');
const trainProgressFill = document.getElementById('trainProgressFill');
const lossText = document.getElementById('lossText');

const uploadTest = document.getElementById('uploadTest');
const btnTest = document.getElementById('btnTest');
const runInferenceBtn = document.getElementById('runInferenceBtn');
const canvasInput = document.getElementById('canvasInput');
const networkVisualizer = document.getElementById('networkVisualizer');

const outputContainers = {
  conv: document.getElementById('convOutputs'),
  pool: document.getElementById('poolOutputs'),
  class: document.getElementById('classOutputs')
};

// ============================================================
// PASSO 0: Carregar o MobileNet ao iniciar a página
// ============================================================
async function loadMobileNet() {
  trainingStatus.textContent = 'Carregando MobileNet pré-treinado...';
  try {
    const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
    );
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    mobileNet = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });

    trainingStatus.textContent = 'MobileNet pronto! Envie suas imagens nas quatro classes.';
    trainingStatus.style.color = '#34d399';
    
    // Ativa os botões de upload de dados
    uiElements.forEach(el => el.btn.disabled = false);
  } catch (error) {
    trainingStatus.textContent = 'Erro ao carregar o MobileNet. Verifique sua conexão.';
    trainingStatus.style.color = '#ef4444';
  }
}
loadMobileNet();

// ============================================================
// PRÉ-PROCESSAMENTO
// ============================================================
function preprocessImage(img) {
  return tf.tidy(() => {
    return tf.browser.fromPixels(img)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
      .toFloat()
      .div(127.5)
      .sub(1) // Normalização para o padrão do MobileNet (-1 a 1)
      .expandDims(0);
  });
}

function extractFeatures(img) {
  return tf.tidy(() => {
    const preprocessed = preprocessImage(img);
    return mobileNet.predict(preprocessed); // Gera o shape [1, 7, 7, 256]
  });
}

// ============================================================
// COLETA DE DADOS (Configurando os 4 botões)
// ============================================================
function setupUploads() {
  uiElements.forEach((el, index) => {
    el.btn.addEventListener('click', () => el.upload.click());

    el.upload.addEventListener('change', async (e) => {
      const files = e.target.files;
      for (let file of files) {
        const img = await loadImage(file);
        classImages[index].push(img);

        const embedding = extractFeatures(img);
        classEmbeddings[index].push(embedding);

        const previewNode = document.createElement('img');
        previewNode.src = URL.createObjectURL(file);
        previewNode.className = 'image-preview';
        el.preview.appendChild(previewNode);
      }
      el.count.textContent = `${classImages[index].length} imagens carregadas`;
      checkReadyToTrain();
    });
  });
}
setupUploads();

function loadImage(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

function checkReadyToTrain() {
  // Verifica se todas as 4 classes possuem pelo menos uma imagem enviada
  const allLoaded = classImages.every(arr => arr.length > 0);
  if (allLoaded) {
    trainBtn.disabled = false;
    trainingStatus.textContent = 'Pronto! Imagens detectadas em todas as categorias. Pode iniciar o treino.';
    trainingStatus.style.color = '#34d399';
  } else {
    trainBtn.disabled = true;
    trainingStatus.textContent = 'Envie pelo menos 1 imagem em CADA uma das 4 classes para liberar o treino.';
    trainingStatus.style.color = '#fbbf24';
  }
}

// ============================================================
// TREINAMENTO MULTI-CLASSE (4 SAÍDAS)
// ============================================================
trainBtn.addEventListener('click', async () => {
  trainBtn.disabled = true;
  uiElements.forEach(el => el.btn.disabled = true);
  trainingStatus.textContent = 'Construindo classificador de 4 classes...';
  trainingStatus.style.color = '#f8fafc';

  const sampleShape = classEmbeddings[0][0].shape.slice(1); // [7, 7, 256]

  classifier = tf.sequential();
  classifier.add(tf.layers.flatten({ inputShape: sampleShape }));
  classifier.add(tf.layers.dense({ units: 64, activation: 'relu', name: 'hidden_1' }));
  classifier.add(tf.layers.dropout({ rate: 0.3 }));
  classifier.add(tf.layers.dense({ units: 4, name: 'dense_out' })); // Alterado para 4 saídas logits
  classifier.add(tf.layers.softmax({ name: 'softmax_out' }));

  classifier.compile({
    optimizer: tf.train.adam(0.0005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  // Montagem dinâmica dos dados e das labels correspondentes (One-Hot Encoding)
  let allEmbeddings = [];
  let labels = [];

  classEmbeddings.forEach((embeddingsArr, classIdx) => {
    embeddingsArr.forEach(embedding => {
      allEmbeddings.push(embedding);
      
      // Cria a label no formato One-Hot (Ex: Classe 2 vira [0, 1, 0, 0])
      let oneHot = [0, 0, 0, 0];
      oneHot[classIdx] = 1;
      labels.push(oneHot);
    });
  });

  const xs = tf.tidy(() => tf.concat(allEmbeddings.map(e => e), 0));
  const ys = tf.tensor2d(labels, [labels.length, 4]);

  trainProgress.classList.remove('hidden');
  lossText.classList.remove('hidden');
  trainingStatus.textContent = 'Treinando classificador Convolucional...';

  const epochs = 150;
  await classifier.fit(xs, ys, {
    epochs,
    batchSize: Math.max(1, Math.floor(allEmbeddings.length / 2)),
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const percent = ((epoch + 1) / epochs) * 100;
        trainProgressFill.style.width = `${percent}%`;
        lossText.textContent = `Época ${epoch + 1}/${epochs} | Loss: ${logs.loss.toFixed(4)} | Acurácia: ${(logs.acc * 100).toFixed(1)}%`;
      }
    }
  });

  xs.dispose();
  ys.dispose();

  trainingStatus.textContent = '✅ Treinamento concluído!';
  trainingStatus.style.color = '#34d399';
  btnTest.disabled = false;
});

// ============================================================
// INFERÊNCIA
// ============================================================
btnTest.addEventListener('click', () => uploadTest.click());

uploadTest.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = await loadImage(file);

  const ctx = canvasInput.getContext('2d');
  ctx.clearRect(0, 0, canvasInput.width, canvasInput.height);
  const scale = Math.min(canvasInput.width / img.width, canvasInput.height / img.height);
  const w = img.width * scale;
  const h = img.height * scale;
  ctx.drawImage(img, (canvasInput.width - w) / 2, (canvasInput.height - h) / 2, w, h);

  if (testImageTensor) testImageTensor.dispose();
  testImageTensor = preprocessImage(img);

  runInferenceBtn.disabled = false;
  networkVisualizer.classList.add('hidden');
});

runInferenceBtn.addEventListener('click', async () => {
  if (!classifier || !testImageTensor) return;
  runInferenceBtn.disabled = true;
  networkVisualizer.classList.remove('hidden');

  const testEmbedding = mobileNet.predict(testImageTensor);

  const denseOutLayer = classifier.getLayer('dense_out');
  const logitsModel = tf.model({ inputs: classifier.inputs, outputs: denseOutLayer.output });

  const logitsTensor = logitsModel.predict(testEmbedding);
  const finalOut = classifier.predict(testEmbedding);

  const logitsData = await logitsTensor.data();
  const probsData = await finalOut.data();

  // Visualizar as saídas das primeiras camadas da CNN
  await renderActivationMaps(testEmbedding, outputContainers.conv, 4);

  const pooled = tf.tidy(() => tf.maxPool(testEmbedding, 2, 2, 'valid'));
  await renderActivationMaps(pooled, outputContainers.pool, 4);
  pooled.dispose();

  await renderClassBars(finalOut);
  renderMathExplanation(logitsData, probsData);

  tf.dispose([testEmbedding, logitsTensor, finalOut]);
  runInferenceBtn.disabled = false;
});

// ============================================================
// RENDERIZAÇÃO DAS ACTIVATION MAPS E BARRAS
// ============================================================
async function renderActivationMaps(tensor, container, maxFilters = 4) {
  container.innerHTML = '';
  const shape = tensor.shape;
  const numFilters = Math.min(shape[3], maxFilters);

  const min = tensor.min();
  const max = tensor.max();
  const normalized = tensor.sub(min).div(max.sub(min).add(1e-5));
  const unstacked = tf.unstack(normalized, 3);

  for (let i = 0; i < numFilters; i++) {
    const filterTensor = unstacked[i].squeeze([0]);
    const rgbTensor = filterTensor.mul(255).cast('int32').expandDims(-1).tile([1, 1, 3]);
    const canvas = document.createElement('canvas');
    canvas.width = shape[2];
    canvas.height = shape[1];
    await tf.browser.toPixels(rgbTensor, canvas);

    const wrap = document.createElement('div');
    wrap.className = 'canvas-wrapper';
    wrap.appendChild(canvas);
    container.appendChild(wrap);

    rgbTensor.dispose();
    filterTensor.dispose();
  }
  normalized.dispose();
  min.dispose();
  max.dispose();
}

async function renderClassBars(probabilitiesTensor) {
  const container = outputContainers.class;
  container.innerHTML = '';
  const probs = await probabilitiesTensor.data();

  for (let i = 0; i < 4; i++) {
    const p = (probs[i] * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'class-bar-row';
    row.innerHTML = `
      <div class="class-label" style="width: 180px; text-align: left;">${classNames[i]}</div>
      <div class="class-track">
        <div class="class-fill" style="width: ${p}%"></div>
      </div>
      <div class="class-prob">${p}%</div>
    `;
    container.appendChild(row);
  }
}

function renderMathExplanation(logits, probs) {
  const explanationBox = document.getElementById('softmaxExplanation');
  const mathSteps = document.getElementById('mathSteps');
  const mathConclusion = document.getElementById('mathConclusion');
  explanationBox.classList.remove('hidden');

  const formatNum = (v) => (Math.abs(v) > 9999 ? v.toExponential(2) : v.toFixed(4));
  
  // Realiza o cálculo matemático exponencial para as 4 classes
  let exps = logits.map(l => Math.exp(l));
  let sumE = exps.reduce((a, b) => a + b, 0);

  let stepsHtml = `
    <li>
      <strong>Passo 1 — Exponenciação dos Logits:</strong><br>
      A rede gerou pontuações brutas para as 4 classes.<br>
      ${classNames.map((name, idx) => `${name}: e<sup>${formatNum(logits[idx])}</sup> = ${formatNum(exps[idx])}`).join('<br>')}
    </li>
    <li>
      <strong>Passo 2 — Soma dos denominadores (Σ):</strong><br>
      Soma de todos os valores e<sup>x</sup> = <strong>${formatNum(sumE)}</strong>
    </li>
    <li>
      <strong>Passo 3 — Probabilidade final (Divisão):</strong><br>
      ${classNames.map((name, idx) => `${name}: ${formatNum(exps[idx])} ÷ ${formatNum(sumE)} = <strong>${(probs[idx] * 100).toFixed(1)}%</strong>`).join('<br>')}
    </li>
  `;
  
  mathSteps.innerHTML = stepsHtml;

  // Encontra qual classe ganhou a maior pontuação
  let winnerIdx = 0;
  let maxProb = 0;
  probs.forEach((p, idx) => {
    if (p > maxProb) {
      maxProb = p;
      winnerIdx = idx;
    }
  });

  const winnerProb = (probs[winnerIdx] * 100).toFixed(1);
  mathConclusion.innerHTML = `Resultado: O modelo possui <strong>${winnerProb}%</strong> de confiança de que a imagem de teste pertence à categoria: <strong>${classNames[winnerIdx]}</strong>.`;
}