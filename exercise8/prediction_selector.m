% prediction_selector.m

% this is a script that pages through all of the movies in the database, ex8

p=prediction_selector(X,Theta,num_movies)
rating=zeros(size(p(:,1)));

% scroll through the list of movies in the database and make a rating.
% rate 1-5 if you have seen the movie, and rate 0 if you have not seen.
fprintf('\nYou will scroll through a list of movies and make ratings\n');
fprintf('for movies. \n');
fprintf('Rate 1-5 if you have seen the movie, rate 0 if you have not seen it.');
fprintf('\nPress enter to continue')
pause
for index=1:num_movies
  prompt=strcat('What is your rating of the movie:',' ', movieList{index},'?');
  rating(index)=input(prompt);  

end

% for any movies rated "0" (so, unwatched movies), add the mean.
% otherwise, matrix elements are included.

end

p = X * Theta';
my_predictions = p(:,1) + Ymean;