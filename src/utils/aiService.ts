import { Character, StoryNode, Choice, Consequence } from '../models/types';

// Simple prompt template for story generation
const generateStoryPrompt = (
  character: Character,
  currentNode: StoryNode,
  choice: Choice,
  history: StoryNode[]
) => {
  return `
You are a creative storyteller in a fantasy adventure game. Continue the story based on the following:

CHARACTER INFO:
Name: ${character.name}
Background: ${character.background}
Health: ${character.attributes.health}
Strength: ${character.attributes.strength}
Intelligence: ${character.attributes.intelligence}
Charisma: ${character.attributes.charisma}

CURRENT SITUATION:
${currentNode.content}

THE PLAYER CHOSE:
${choice.text}

RECENT HISTORY:
${history.slice(-3).map(node => node.content).join('\n\n')}

GUIDELINES:
1. Write in third person perspective, focusing on ${character.name}'s actions and experiences.
2. Continue the story with 2-3 paragraphs.
3. Make the consequences of the player's choice clear.
4. End with a situation where the player needs to make a new choice.
5. Provide 3 distinct choices that branch the story in different directions.
6. Make sure choices have meaningful consequences.
7. Keep the tone consistent with the story so far.

RESPONSE FORMAT:
{
  "content": "The continuation of the story...",
  "choices": [
    {"text": "First choice..."},
    {"text": "Second choice..."},
    {"text": "Third choice..."}
  ]
}
`;
};

// Simple prompt template for image generation
const generateImagePrompt = (storyNode: StoryNode) => {
  return `
Generate a simple image for the following scene in a fantasy adventure:

${storyNode.content}

Make it a simple fantasy style illustration.
`;
};

// Improved story generation with error handling and retry
export const generateStoryText = async (
  character: Character,
  currentNode: StoryNode | null,
  choice: Choice | null,
  history: StoryNode[]
): Promise<{ content: string; choices: { text: string; consequence?: Consequence }[] }> => {
  try {
    console.log('Calling backend API for story generation...');
    const response = await fetch('/generate-story', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        character,
        current_node: currentNode,
        choice,
        history,
        story_start: history.length === 0  // Set story_start to true when there's no history
      })
    });
    
    if (!response.ok) {
      console.error(`Error from server: ${response.status} ${response.statusText}`);
      throw new Error(`Server returned ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('Story generated successfully:', data);
    return { content: data.content, choices: data.choices };
  } catch (error) {
    console.error('Failed to generate story:', error);
    // Use fallback story generation
    return fallbackStoryGeneration(character, currentNode, choice);
  }
};

// Update fallback story generation to remove inventory references
const fallbackStoryGeneration = (
  character: Character, 
  currentNode: StoryNode | null, 
  choice: Choice | null
): { content: string; choices: { text: string; consequence?: Consequence }[] } => {
  // For initial story (when currentNode and choice are null)
  if (!currentNode || !choice) {
    return {
      content: `${character.name} begins their adventure in a world of mystery and magic. With ${character.background}, they step forward into the unknown.`,
      choices: [
        { 
          text: "Explore the nearby forest",
          consequence: { attribute: { name: 'intelligence', change: 1 } }
        },
        { 
          text: "Head towards the village",
          consequence: { attribute: { name: 'charisma', change: 1 } }
        },
        { 
          text: "Rest and prepare for the journey ahead",
          consequence: { attribute: { name: 'health', change: 5 } }
        }
      ]
    };
  }

  // For continuing story
  const choiceText = choice.text.toLowerCase();
  let content = `${character.name} ${choiceText}. The path ahead reveals new possibilities.`;
  
  return {
    content,
    choices: [
      { 
        text: "Investigate further", 
        consequence: { attribute: { name: 'intelligence', change: 1 } }
      },
      { 
        text: "Proceed with caution",
        consequence: { attribute: { name: 'health', change: 5 } }
      },
      { 
        text: "Try a different approach",
        consequence: { attribute: { name: 'strength', change: 1 } }
      }
    ]
  };
};

// Improved image generation with error handling
export const generateImage = async (storyNode: StoryNode): Promise<string> => {
  try {
    console.log('Calling backend API for image generation...');
    const response = await fetch('/generate-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ story_text: storyNode.content })
    });

    if (!response.ok) {
      console.error(`Image API returned status ${response.status} ${response.statusText}`);
      throw new Error(`Image API error: ${response.status}`);
    }

    const result = await response.json();
    console.log('Image API response:', result);

    // Fallback to Unsplash if generation was not successful
    if (!result.success) {
      const keywords = extractKeywords(storyNode.content);
      console.warn('Image API fallback, using Unsplash:', keywords);
      return `https://source.unsplash.com/random/800x500/?fantasy,${keywords}`;
    }

    // Build absolute URL for the generated image
    const imageUrl = result.image_url.startsWith('http')
      ? result.image_url
      : `${window.location.origin}${result.image_url}`;

    return imageUrl;
  } catch (error) {
    console.error('Error generating image:', error);
    const keywords = extractKeywords(storyNode.content);
    return `https://source.unsplash.com/random/800x500/?fantasy,${keywords}`;
  }
};

// Helper to extract keywords for fallback image
const extractKeywords = (text: string): string => {
  const common = new Set(['the','and','for','with','that','they','this','then','into','your','their']);
  const words = text.toLowerCase().match(/\b\w+\b/g) || [];
  const keywords = words
    .filter(word => word.length > 3 && !common.has(word))
    .slice(0, 3)
    .join(',');
  return keywords || 'fantasy';
};

// Function to initialize AI models (would load local models in a real implementation)
export const initializeAI = async (): Promise<void> => {
  console.log('Initializing AI models...');
  // In a real implementation, this would load local AI models
  await new Promise(resolve => setTimeout(resolve, 1000));
  console.log('AI models initialized');
}; 