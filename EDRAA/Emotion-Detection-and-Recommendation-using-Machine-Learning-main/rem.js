// Function to show the card form for adding a new task
function showCardForm(columnId) {
    var column = document.getElementById(columnId);
    var cardsContainer = column.querySelector('.cards');
  
    // Create a new card form element
    var cardForm = document.createElement('div');
    cardForm.classList.add('card', 'add-card-form');
    cardForm.innerHTML = `
      <input type="text" placeholder="Enter task">
      <button onclick="addCard('${columnId}')">Add</button>
    `;
    
    // Append the card form to the cards container
    cardsContainer.appendChild(cardForm);
  }
  
  // Function to add a new card to the specified column
  function addCard(columnId) {
    var column = document.getElementById(columnId);
    var cardsContainer = column.querySelector('.cards');
    var input = cardsContainer.querySelector('input');
    var taskName = input.value.trim();
  
    // Ensure task name is not empty
    if (taskName !== '') {
      // Create a new card element
      var card = document.createElement('div');
      card.classList.add('card');
      card.draggable = true;
      card.textContent = taskName;
  
      // Append the card to the cards container
      cardsContainer.insertBefore(card, cardsContainer.lastElementChild);
      
      // Clear the input field
      input.value = '';
    }
  }
  
  // Function to allow dropping cards into the column
  function allowDrop(event) {
    event.preventDefault();
  }
  
  // Function to handle dropping cards into the column
  function drop(event) {
    event.preventDefault();
    var data = event.dataTransfer.getData('text');
    event.target.appendChild(document.getElementById(data));
  }
  
  // Link the script to the add-card buttons
  document.addEventListener('DOMContentLoaded', function() {
    var addCardButtons = document.querySelectorAll('.add-card');
    addCardButtons.forEach(function(button) {
      button.addEventListener('click', function() {
        var columnId = button.parentNode.id;
        showCardForm(columnId);
      });
    });
  });
  