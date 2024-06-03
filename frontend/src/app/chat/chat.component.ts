import { Component, Input } from '@angular/core';
import { ChatService } from '../chat.service';
import { CommonModule, NgIf } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html', // <- Ensure this path is correct
  styleUrls: ['./chat.component.scss'],
  standalone: true,
  imports: [NgIf, FormsModule, CommonModule],

})
export class ChatComponent {
  userInput: string = '';
  @Input() chat=''
  messages4: { sender: string, text: string , score: number}[] = [
    { sender: 'bot', text: 'Hello! How can I assist you today?' , score: 0}
  ];
  messages3: { sender: string, text: string ,score: number }[] = [
    { sender: 'bot', text: 'Hello! How can I assist you today?',  score: 0 }
  ];
  messagesL: { sender: string, text: string ,score: number }[] = [
    { sender: 'bot', text: 'Hello! How can I assist you today?',  score: 0 }
  ];

  constructor(private chatService: ChatService) {}

  async sendMessage() {
    if (this.userInput.trim() === '') {
      return;
    }
  
    // Add user message to messages array
    this.messages4.push({ sender: 'user', text: this.userInput ,score: 0});
    this.messages3.push({ sender: 'user', text: this.userInput, score: 0 });
    this.messagesL.push({ sender: 'user', text: this.userInput, score: 0 });
  
    // Store the user input locally
    const userMessage = this.userInput;
    
    // Clear the input field
    this.userInput = '';
  
    try {
      // Send request to Chatbot 4
      const response4 = await this.chatService.sendMessage("4"+'/'+userMessage);
      // Add bot response to messages array
      this.messages4.push({ sender: 'bot', text: response4.response, score: response4.score});
  
      // Send request to Chatbot 3
      const response3 = await this.chatService.sendMessage("3"+'/'+userMessage);
      // Add bot response to messages array
      this.messages3.push({ sender: 'bot', text: response3.response , score: response3.score});
      const responseL = await this.chatService.sendMessage("L"+'/'+userMessage);
      // Add bot response to messages array
      this.messagesL.push({ sender: 'bot', text: responseL.response , score: responseL.score});
    } catch (error) {
      console.error(error);
    }
  }
}