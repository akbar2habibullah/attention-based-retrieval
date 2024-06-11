import cosineSimilarity from 'compute-cosine-similarity'
import { GoogleGenerativeAI } from "@google/generative-ai"
import Groq from "groq-sdk"

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY })

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GEMINI_API_KEY)
const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" })


// Function to generate response from Groq (can be from anywhere, e.g, OpenAI, OpenRouter, Gemini, Claude, Mistral, etc)
export async function generateResponse(prompt, model = 'llama3-8b-8192') {
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
export async function findMostInfluentialTokens(basePrompt, threshold = 0.35) {
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
    const similarity = cosineSimilarity(baseVec, variantVec) || 0
    return 1 - similarity
  })

  // Find the maximum distance
  const maxDistance = Math.max(...distances)

  // Determine the threshold distance
  const thresholdDistance = maxDistance * (1 - threshold)

  // Identify influential tokens based on the threshold
  const words = basePrompt.split(' ')
  const influentialTokens = words.filter((_, index) => distances[index] >= thresholdDistance)

  console.log(`Influential tokens: ${influentialTokens}`)

  return influentialTokens
}

const result = await findMostInfluentialTokens("What are the benefits of a balanced diet and regular exercise for overall health?")

console.log(result)
