import cosineSimilarity from 'compute-cosine-similarity'
import { GoogleGenerativeAI } from "@google/generative-ai"
import Groq from "groq-sdk"

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY })

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GEMINI_API_KEY)
const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" })


// Function to generate response from Groq (can be from anywhere, e.g, OpenAI, OpenRouter, Gemini, Claude, Mistral, etc)
export async function generateResponse(prompt, model = 'llama3-70b-8192') {
  const chatCompletion = await groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
    model,
  })

  const result = chatCompletion.choices[0]?.message?.content
  return result || ""
}

// Function to get embedding from Google text embedding
async function getEmbedding(text) {
  const result = await embeddingModel.embedContent(text)
  return result.embedding.values
}

// Function to create prompt variants
function createVariants(prompt) {
  const words = prompt.split(' ')
  return words.map((word, index) => {
    const replaced = words.slice()
    replaced[index] = '<placeholder>'
    return replaced.join(' ')
  })
}

// Main function to find the most influential tokens
async function findMostInfluentialTokens(basePrompt, threshold = 0.15) {
  // Step 1: Generate base response and embedding
  const baseResponse = await generateResponse(basePrompt)
  const baseEmbedding = await getEmbedding(baseResponse)

  // Step 2: Create variants of the prompt
  const variants = createVariants(basePrompt)

  // Step 3: Generate outputs and embeddings for variants
  const variantOutputs = await Promise.all(variants.map(str => generateResponse(str)))
  const variantEmbeddings = await Promise.all(variantOutputs.map(str => getEmbedding(str)))

  // Step 4: Compute embedding distances
  const baseVec = baseEmbedding
  const distances = variantEmbeddings.map(variantVec => {
    const similarity = cosineSimilarity(baseVec, variantVec)
    return 1 - similarity
  })

  // Step 5: Find the three most influential tokens
  const sortedIndices = distances
    .map((distance, index) => ({ distance, index }))
    .sort((a, b) => b.distance - a.distance)
    .slice(0, 3)
    .map(item => item.index)

  const maxDistances = sortedIndices.map(index => distances[index])

  // Determine the threshold distance for each of the top three tokens
  const thresholdDistances = maxDistances.map(distance => distance * (1 - threshold))

  // Identify influential tokens based on the threshold
  const words = basePrompt.split(' ')
  const influentialTokens = new Set()

  sortedIndices.forEach((topIndex, i) => {
    influentialTokens.add(words[topIndex])
    words.forEach((word, index) => {
      if (distances[index] >= thresholdDistances[i]) {
        influentialTokens.add(word)
      }
    })
  })

  const orderedInfluentialTokens = words.filter(word => influentialTokens.has(word))

  return orderedInfluentialTokens
}

function splitIntoSentences(paragraph) {
  return paragraph.match(/[^.!?]+[.!?]+/g) || [paragraph]
}

// Main function to process each sentence in a paragraph
async function findInfluentialTokensInParagraph(paragraph) {
  const sentences = splitIntoSentences(paragraph)
  const influentialTokensArray = await Promise.all(sentences.map(sentence => findInfluentialTokensForSentence(sentence)))
  console.log(`Influential tokens array: ${JSON.stringify(influentialTokensArray, null, 2)}`)
}

const arr = [
  "What are the benefits of a balanced diet and regular exercise for overall health?",
  "Describe the process of photosynthesis and its importance to plant life.",
  "Do you remember the name of the book I mentioned last week about artificial intelligence?",
  "Can you recall the recipe for the chocolate cake we made during the holidays?",
  "How can I reset my password if I have forgotten my current one?",
  "What are the shipping options available for international orders?",
  "What are the common causes of a computer overheating and how can it be prevented?",
  "Explain how to set up a virtual private network (VPN) on a home router.",
  "Remind me to schedule a meeting with the project team on Monday at 10 AM.",
  "Can you find a good Italian restaurant near me for dinner tonight?",
]

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

for (let i = 0; i < arr.length; i++) {
  const result = await findMostInfluentialTokens(arr[i])

  console.log(`Prompt: ${arr[i]}`)
  console.log(`Influential tokens: ${result}`)
  console.log()

  await delay(30000)
}