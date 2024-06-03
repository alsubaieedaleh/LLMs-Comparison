import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private apiUrl = 'http://127.0.0.1:5000/Chat';

  constructor() { }

  async sendMessage(query: string): Promise<any> {
    return await fetch(this.apiUrl + query, {
      method: 'POST',

      headers: {
        'Content-Type': 'application/json'
      }
    })
    .then(response => {
      if (!response.ok) {
        throw `Server error: [${response.status}] [${response.statusText}] [${response.url}]`;
      }
      console.log(response);
      return response.json();
    })
    .then(receivedJson => {
      // your code with json here...
      return receivedJson;
    })
    .catch(err => {
      console.debug("Error in fetch", err);
      // setErrors(err) // Uncomment this line if you have a setErrors function
    });
  }
}