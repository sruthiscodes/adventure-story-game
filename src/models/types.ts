export interface Character {
  name: string;
  attributes: {
    health: number;
    strength: number;
    intelligence: number;
    charisma: number;
  };
  background: string;
}

export interface GameState {
  character: Character;
  currentStoryNode: StoryNode;
  storyHistory: StoryNode[];
  gameStarted: boolean;
  currentImage: string | null;
}

export interface StoryNode {
  id: string;
  content: string;
  choices: Choice[];
  image?: string;
  timestamp: number;
}

export interface Choice {
  id: string;
  text: string;
  consequence?: Consequence;
  nextNodeId?: string;
}

export interface Consequence {
  attribute?: {
    name: 'health' | 'strength' | 'intelligence' | 'charisma';
    change: number;
  };
} 