import React, { useState } from 'react';
import styled from '@emotion/styled';
import { Character } from '../models/types';
import { motion } from 'framer-motion';

interface CharacterCreationProps {
  onCreateCharacter: (character: Character) => void;
}

const backgroundOptions = [
  'A wandering adventurer seeking fortune and glory.',
  'A disgraced noble in search of redemption.',
  'A cunning thief with a secret mission.',
  'A powerful mage studying the ancient arts.',
  'A skilled ranger who protects the wilderness.',
  'A holy warrior championing a righteous cause.',
  'A village healer with mysterious powers.',
  'An exiled warrior from a fallen kingdom.',
  'A scholar seeking forgotten knowledge.',
  'A merchant who stumbled upon dark secrets.',
  'A shipwrecked sailor far from home.',
  'A cursed individual seeking to break their fate.',
];

const CharacterCreation: React.FC<CharacterCreationProps> = ({ onCreateCharacter }) => {
  const [character, setCharacter] = useState<Character>({
    name: '',
    attributes: {
      health: 100,
      strength: 10,
      intelligence: 10,
      charisma: 10,
    },
    background: 'A wandering adventurer seeking fortune and glory.',
  });

  const [attributePoints, setAttributePoints] = useState(5);
  const [nameError, setNameError] = useState('');

  // Add custom background state
  const [customBackgroundEnabled, setCustomBackgroundEnabled] = useState(false);
  const [customBackground, setCustomBackground] = useState('');

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCharacter({ ...character, name: e.target.value });
    if (e.target.value.trim() === '') {
      setNameError('Name is required');
    } else {
      setNameError('');
    }
  };

  const handleBackgroundChange = (option: string) => {
    // If using custom background, don't set it here
    if (customBackgroundEnabled) return;
    
    setCharacter({ ...character, background: option });
  };

  const handleCustomBackgroundChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCustomBackground(e.target.value);
    setCharacter({ ...character, background: e.target.value });
  };

  const toggleCustomBackground = () => {
    setCustomBackgroundEnabled(!customBackgroundEnabled);
    
    // When switching to predefined, select the first background option
    if (customBackgroundEnabled) {
      setCharacter({ ...character, background: backgroundOptions[0] });
    } else {
      // When switching to custom, use the current custom text or provide a starter
      setCharacter({ 
        ...character, 
        background: customBackground || 'Write your character\'s unique background story here...' 
      });
    }
  };

  const handleAttributeChange = (attribute: string, value: number) => {
    // Check if we have enough points to increase
    if (value > 0 && attributePoints <= 0) return;

    // Check if attribute can be decreased (minimum value is 5)
    if (value < 0 && character.attributes[attribute as keyof typeof character.attributes] <= 5) return;

    setAttributePoints(prev => prev - value);
    setCharacter({
      ...character,
      attributes: {
        ...character.attributes,
        [attribute]: character.attributes[attribute as keyof typeof character.attributes] + value
      }
    });
  };

  const handleRandomize = () => {
    // Reset attribute points
    setAttributePoints(5);

    // Generate random attributes (between 5 and 15)
    const strength = Math.floor(Math.random() * 11) + 5;
    const intelligence = Math.floor(Math.random() * 11) + 5;
    const charisma = Math.floor(Math.random() * 11) + 5;

    // Choose random background
    const randomBackground = backgroundOptions[Math.floor(Math.random() * backgroundOptions.length)];

    setCharacter({
      ...character,
      attributes: {
        health: 100,
        strength,
        intelligence,
        charisma
      },
      background: randomBackground
    });
  };

  const handleSubmit = () => {
    // Clear previous error
    setNameError('');
    
    // Validate required fields
    if (!character.name || character.name.trim() === '') {
      setNameError('Name is required');
      return;
    }
    
    // Create a complete character with all required fields
    const completeCharacter: Character = {
      name: character.name.trim(),
      attributes: {
        health: character.attributes.health,
        strength: character.attributes.strength,
        intelligence: character.attributes.intelligence,
        charisma: character.attributes.charisma
      },
      background: character.background || backgroundOptions[0]
    };
    
    try {
      console.log('Submitting character:', completeCharacter);
      onCreateCharacter(completeCharacter);
    } catch (error) {
      console.error('Error creating character:', error);
      setNameError('Error creating character. Please try again.');
    }
  };

  return (
    <Container
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Title>Create Your Character</Title>
      
      <FormGroup>
        <Label>Character Name</Label>
        <Input 
          type="text" 
          value={character.name} 
          onChange={handleNameChange}
          placeholder="Enter character name"
        />
        {nameError && <ErrorText>{nameError}</ErrorText>}
      </FormGroup>

      <Section>
        <SectionTitle>Character Background</SectionTitle>
        
        <BackgroundToggle>
          <ToggleButton 
            selected={!customBackgroundEnabled}
            onClick={() => toggleCustomBackground()}
          >
            Select Preset Background
          </ToggleButton>
          <ToggleButton 
            selected={customBackgroundEnabled}
            onClick={() => toggleCustomBackground()}
          >
            Create Custom Background
          </ToggleButton>
        </BackgroundToggle>
        
        {customBackgroundEnabled ? (
          <CustomBackgroundInput
            value={customBackground || character.background}
            onChange={handleCustomBackgroundChange}
            placeholder="Write your character's unique background story here..."
            rows={4}
          />
        ) : (
          <BackgroundOptions>
            {backgroundOptions.map((option) => (
              <BackgroundOption
                key={option}
                selected={character.background === option}
                onClick={() => handleBackgroundChange(option)}
              >
                {option}
              </BackgroundOption>
            ))}
          </BackgroundOptions>
        )}
      </Section>

      <AttributesContainer>
        <Label>Attributes (Points Remaining: {attributePoints})</Label>
        
        <AttributeGroup>
          <AttributeLabel>Strength: {character.attributes.strength}</AttributeLabel>
          <ButtonGroup>
            <AttributeButton 
              onClick={() => handleAttributeChange('strength', -1)}
              disabled={character.attributes.strength <= 5}
            >
              -
            </AttributeButton>
            <AttributeButton 
              onClick={() => handleAttributeChange('strength', 1)}
              disabled={attributePoints <= 0}
            >
              +
            </AttributeButton>
          </ButtonGroup>
        </AttributeGroup>
        
        <AttributeGroup>
          <AttributeLabel>Intelligence: {character.attributes.intelligence}</AttributeLabel>
          <ButtonGroup>
            <AttributeButton 
              onClick={() => handleAttributeChange('intelligence', -1)}
              disabled={character.attributes.intelligence <= 5}
            >
              -
            </AttributeButton>
            <AttributeButton 
              onClick={() => handleAttributeChange('intelligence', 1)}
              disabled={attributePoints <= 0}
            >
              +
            </AttributeButton>
          </ButtonGroup>
        </AttributeGroup>
        
        <AttributeGroup>
          <AttributeLabel>Charisma: {character.attributes.charisma}</AttributeLabel>
          <ButtonGroup>
            <AttributeButton 
              onClick={() => handleAttributeChange('charisma', -1)}
              disabled={character.attributes.charisma <= 5}
            >
              -
            </AttributeButton>
            <AttributeButton 
              onClick={() => handleAttributeChange('charisma', 1)}
              disabled={attributePoints <= 0}
            >
              +
            </AttributeButton>
          </ButtonGroup>
        </AttributeGroup>
      </AttributesContainer>

      <ButtonContainer>
        <RandomizeButton onClick={handleRandomize}>Randomize</RandomizeButton>
        <StartButton onClick={handleSubmit}>Begin Adventure</StartButton>
      </ButtonContainer>
    </Container>
  );
};

const Container = styled(motion.div)`
  max-width: 600px;
  margin: 0 auto;
  padding: 2rem;
  background-color: rgba(0, 0, 0, 0.8);
  border-radius: 10px;
  color: #f0f0f0;
  position: relative;
  z-index: 1;
`;

const Title = styled.h2`
  text-align: center;
  margin-bottom: 2rem;
  color: #ffd700;
`;

const FormGroup = styled.div`
  margin-bottom: 1.5rem;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 0.5rem;
  font-weight: bold;
  color: #ffd700;
`;

const Input = styled.input`
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #444;
  background-color: #333;
  color: #f0f0f0;
  border-radius: 4px;
  font-size: 1rem;
`;

const Section = styled.div`
  margin-bottom: 1.5rem;
`;

const SectionTitle = styled.h3`
  margin-bottom: 0.5rem;
  color: #ffd700;
`;

const BackgroundToggle = styled.div`
  display: flex;
  margin-bottom: 15px;
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
`;

const ToggleButton = styled.button<{ selected: boolean }>`
  flex: 1;
  padding: 10px;
  background: ${(props) => (props.selected ? '#4a69bd' : '#192a56')};
  color: white;
  border: none;
  font-weight: ${(props) => (props.selected ? 'bold' : 'normal')};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${(props) => (props.selected ? '#4a69bd' : '#273c75')};
  }
`;

const BackgroundOptions = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 200px;
  overflow-y: auto;
  padding-right: 10px;
  
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: #192a56;
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #4a69bd;
    border-radius: 4px;
  }
`;

const BackgroundOption = styled.div<{ selected: boolean }>`
  padding: 10px 15px;
  border-radius: 6px;
  background: ${(props) => (props.selected ? '#4a69bd' : '#192a56')};
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${(props) => (props.selected ? '#4a69bd' : '#273c75')};
  }
`;

const CustomBackgroundInput = styled.textarea`
  width: 100%;
  padding: 12px;
  border-radius: 6px;
  background: #192a56;
  color: white;
  border: 1px solid #4a69bd;
  resize: vertical;
  min-height: 120px;
  font-family: inherit;
  
  &:focus {
    outline: none;
    border-color: #6a89cc;
  }
`;

const AttributesContainer = styled.div`
  margin-bottom: 1.5rem;
`;

const AttributeGroup = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 0.5rem;
  padding: 0.5rem;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
`;

const AttributeLabel = styled.span`
  font-size: 1rem;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 0.5rem;
`;

const AttributeButton = styled.button<{ disabled?: boolean }>`
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: none;
  background-color: ${(props: { disabled?: boolean }) => props.disabled ? '#555' : '#ffd700'};
  color: ${(props: { disabled?: boolean }) => props.disabled ? '#888' : '#000'};
  font-weight: bold;
  cursor: ${(props: { disabled?: boolean }) => props.disabled ? 'not-allowed' : 'pointer'};
`;

const ButtonContainer = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
`;

const RandomizeButton = styled.button`
  padding: 0.75rem 1.5rem;
  background-color: #555;
  color: #f0f0f0;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.2s;
  &:hover {
    background-color: #666;
  }
`;

const StartButton = styled.button`
  flex: 1;
  padding: 0.75rem 1.5rem;
  background-color: #ffd700;
  color: #000;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.2s;
  &:hover {
    background-color: #ffec1f;
  }
`;

const ErrorText = styled.span`
  color: #ff6b6b;
  font-size: 0.85rem;
  margin-top: 0.25rem;
  display: block;
`;

export default CharacterCreation; 