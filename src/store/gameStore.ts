import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { v4 as uuidv4 } from 'uuid';
import { Character, GameState, StoryNode, Choice, Consequence } from '../models/types';
import { generateStoryText, generateImage } from '../utils/aiService';

interface GameStore extends GameState {
  initializeGame: () => void;
  startNewGame: (character: Character) => void;
  makeChoice: (choice: Choice) => Promise<void>;
  saveGame: () => void;
  loadGame: () => boolean;
  clearPersistedState: () => void;
  updateCharacterAttribute: (attributeName: string, value: number) => void;
  setCurrentImage: (imageUrl: string | null) => void;
}

const initialCharacter: Character = {
  name: '',
  attributes: {
    health: 100,
    strength: 10,
    intelligence: 10,
    charisma: 10
  },
  background: ''
};

const initialStoryNode: StoryNode = {
  id: 'initial',
  content: 'Your adventure awaits. Choose your path wisely.',
  choices: [],
  timestamp: Date.now()
};

const initialState: GameState = {
  character: initialCharacter,
  currentStoryNode: initialStoryNode,
  storyHistory: [],
  gameStarted: false,
  currentImage: null
};

export const useGameStore = create<GameStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      initializeGame: () => {
        set(initialState);
      },

      startNewGame: (character: Character) => {
        // First, make sure we're in a clean state
        set({
          ...initialState,
          character,
          gameStarted: true,
          // Initially set currentImage to null to ensure proper loading state
          currentImage: null
        });

        // Generate the initial story
        (async () => {
          try {
            // Generate initial story with story_start flag
            const initialStory = await generateStoryText(
              character,
              null,  // No current node for initial story
              null,  // No choice for initial story
              []     // Empty history
            );

            // Create the initial story node
            const startNode: StoryNode = {
              id: 'start',
              content: initialStory.content,
              choices: initialStory.choices.map(c => ({
                id: uuidv4(),
                text: c.text,
                consequence: c.consequence
              })),
              timestamp: Date.now()
            };

            // Update the state with the generated story
            set({ currentStoryNode: startNode });

            // Generate the initial image
            try {
              const imageUrl = await generateImage(startNode);
              // Only set the image if it was successfully generated
              if (imageUrl) {
                set({ currentImage: imageUrl });
              }
            } catch (error) {
              console.error("Error generating initial image:", error);
              // Set to a fallback image in case of error
              set({ currentImage: 'https://cdn.pixabay.com/photo/2017/08/30/01/05/milky-way-2695569_1280.jpg' });
            }
            
            console.log("Game started successfully!");
          } catch (error) {
            console.error("Error starting game:", error);
          }
        })();
      },

      makeChoice: async (choice: Choice) => {
        const { character, currentStoryNode, storyHistory } = get();
        
        // Apply consequences if any
        if (choice.consequence) {
          applyConsequence(choice.consequence, get, set);
        }
        
        // Clear current image to indicate loading state
        set({ currentImage: null });
        
        // Generate the next story node based on the choice
        const nextNode = await generateNextStoryNode(choice, character, currentStoryNode, storyHistory);
        
        // Update game state first to show the new text
        set((state) => ({
          currentStoryNode: nextNode,
          storyHistory: [...state.storyHistory, state.currentStoryNode]
        }));
        
        // Then generate image for the new node
        try {
          const imageUrl = await generateImage(nextNode);
          // Only set image if it was successfully generated
          if (imageUrl) {
            set({ currentImage: imageUrl });
          } else {
            // Use a fallback image if nothing was returned
            set({ currentImage: 'https://cdn.pixabay.com/photo/2017/08/30/01/05/milky-way-2695569_1280.jpg' });
          }
        } catch (error) {
          console.error('Error generating image:', error);
          // Use a fallback image
          set({ currentImage: 'https://cdn.pixabay.com/photo/2017/08/30/01/05/milky-way-2695569_1280.jpg' });
        }
      },

      saveGame: () => {
        // Using persist middleware, the state is automatically saved to localStorage
        console.log('Game saved');
      },

      loadGame: () => {
        // Check if there's a valid saved game with actual content
        const state = get();
        const hasValidSave = state.gameStarted && 
                            state.character.name !== '' && 
                            state.storyHistory.length > 0;
        return hasValidSave;
      },

      clearPersistedState: () => {
        localStorage.removeItem('ai-storyteller-storage');
        set(initialState);
      },

      updateCharacterAttribute: (attributeName: string, value: number) => {
        set((state) => ({
          character: {
            ...state.character,
            attributes: {
              ...state.character.attributes,
              [attributeName]: Math.max(0, (state.character.attributes as any)[attributeName] + value)
            }
          }
        }));
      },

      setCurrentImage: (imageUrl: string | null) => {
        set({ currentImage: imageUrl });
      }
    }),
    {
      name: 'ai-storyteller-storage',
      storage: createJSONStorage(() => localStorage)
    }
  )
);

// Helper function to apply consequences of choices
function applyConsequence(
  consequence: Consequence,
  get: () => GameStore,
  set: (partial: Partial<GameStore> | ((state: GameStore) => Partial<GameStore>)) => void
) {
  const store = get();
  
  // Apply attribute changes
  if (consequence.attribute) {
    store.updateCharacterAttribute(
      consequence.attribute.name,
      consequence.attribute.change
    );
  }
}

// Story generation using AI
async function generateNextStoryNode(
  choice: Choice,
  character: Character,
  currentNode: StoryNode,
  storyHistory: StoryNode[]
): Promise<StoryNode> {
  try {
    // Call the AI service to generate the next part of the story
    const aiResponse = await generateStoryText(character, currentNode, choice, storyHistory);
    
    // Create a new story node
    return {
      id: uuidv4(),
      content: aiResponse.content,
      choices: aiResponse.choices.map(c => ({
        id: uuidv4(),
        text: c.text,
        consequence: c.consequence
      })),
      timestamp: Date.now()
    };
  } catch (error) {
    console.error('Error generating next story node:', error);
    // Return a fallback node in case of error
    return {
      id: uuidv4(),
      content: 'Something went wrong with the story generation. Please try again.',
      choices: [
        {
          id: uuidv4(),
          text: 'Try again',
          consequence: undefined
        }
      ],
      timestamp: Date.now()
    };
  }
} 